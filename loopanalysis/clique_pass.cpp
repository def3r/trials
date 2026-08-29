#include <optional>
#include <string>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "clique-cpp"  // -debug-only=clique-cpp

using namespace llvm;

// Data structures
//
// Like kcolor, maximum-clique search is solved by self-recursion, not an
// outer enumeration loop -- but the recursion shape itself differs from
// kcolor's in three structural ways (see analysis/clique/clique.md for the
// IR-verified derivation):
//
//   1. The self-call's "next candidate position" argument advances via the
//      candidate loop's OWN induction variable + 1 (`v + 1`), not via the
//      function's own formal parameter + 1 the way kcolor's `node + 1`
//      does. A second self-call argument ("size") DOES advance via
//      FormalArg + 1, kcolor's style -- both patterns appear in the same
//      self-call, at different argument positions.
//   2. There is no backtrack store: clique[size] is simply overwritten by
//      the next loop iteration, so the matcher must not require one.
//   3. The result is accumulated via TWO std::max call sites updating the
//      same memory-resident accumulator (accept-this-extension, and
//      keep-searching-deeper), with no early exit -- MaxCut's
//      running-best-across-all-iterations shape, not kcolor's
//      early-return-on-first-success shape.
struct MaxCliqueMatch {
  Function* MaxCliquesFn;
  Argument* StartArg;   // next candidate vertex to try; loop starts here
  Argument* CliqueArg;  // clique-membership container, indexed by SizeArg
  Argument* SizeArg;    // current clique size (also the recursion depth)
  Argument* NArg;       // vertex count / loop bound
  Argument* GraphArg;   // adjacency matrix, threaded through unchanged

  Loop* CandidateLoop;   // the "try each remaining vertex" loop
  PHINode* LoopPhi;      // v's phi -- the loop's own induction variable
  CallBase* SelfCall;    // maxCliques(v + 1, clique, size + 1, N, graph)
  CallBase* GuardCall;   // isClique(size + 1, clique, graph)
  Function* GuardFn;

  CallBase* AssignIndex;  // clique[size] operator[] call
  StoreInst* AssignStore;  // clique[size] = v

  AllocaInst* BestAcc;
  CallBase* AcceptMaxCall;  // best = max(best, size + 1)
  CallBase* ExtendMaxCall;  // best = max(best, <self-call result>)
};

// Helpers (mirrors kcolor_pass.cpp; duplicated per that established
// convention of not sharing a common header between passes).

static Value* stripToContainerSource(Value* V) {
  while (V) {
    if (isa<AllocaInst>(V) || isa<Argument>(V))
      return V;
    if (auto* CB = dyn_cast<CallBase>(V)) {
      if (CB->arg_size() == 0)
        return nullptr;
      V = CB->getArgOperand(0);
      continue;
    }
    if (auto* BC = dyn_cast<BitCastInst>(V)) {
      V = BC->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

static Value* stripIntCasts(Value* V) {
  while (auto* CI = dyn_cast<CastInst>(V))
    V = CI->getOperand(0);
  return V;
}

static std::string normalizeDemangled(std::string d) {
  for (auto pos = d.find("[abi:"); pos != std::string::npos;
       pos = d.find("[abi:")) {
    auto end = d.find(']', pos);
    if (end == std::string::npos)
      break;
    d.erase(pos, end - pos + 1);
  }
  const std::string needle = "std::__1::";
  for (auto pos = d.find(needle); pos != std::string::npos;
       pos = d.find(needle))
    d.erase(pos + 5, 5);
  return d;
}

static bool demangleContains(const CallBase* CB, StringRef Sub) {
  const Function* F = CB->getCalledFunction();
  if (!F)
    return false;
  std::string demangled = normalizeDemangled(demangle(F->getName()));
  return StringRef(demangled).contains(Sub);
}

static bool isOperatorBracket(const CallBase* CB) {
  return demangleContains(CB, "operator[]");
}

static Value* getCondBrCondition(BasicBlock* BB) {
  if (auto* CBR = dyn_cast<CondBrInst>(BB->getTerminator()))
    return CBR->getCondition();
  return nullptr;
}

// Returns true iff V is structurally `add nsw/nuw Base, 1`. Deliberately
// structural (never relies on shared SSA identity) -- with the pipeline
// this pass expects (no early-cse), `size + 1` is computed independently
// at each of its three use sites, and each occurrence must be re-derived
// on its own rather than compared by pointer identity.
static bool isAddOne(Value* V, Value* Base) {
  auto* BO = dyn_cast<BinaryOperator>(V);
  if (!BO || BO->getOpcode() != Instruction::Add)
    return false;
  Value *Op0 = BO->getOperand(0), *Op1 = BO->getOperand(1);
  auto* C = dyn_cast<ConstantInt>(Op1);
  return Op0 == Base && C && C->isOne();
}

// Match a `std::max<int>` call site that updates BestAcc from a second,
// temporary-alloca-backed operand, returning the value that was stored
// into that temporary. Mirrors maxcut_pass.cpp's min/max-call recognition
// style rather than kcolor_pass.cpp's single early-return pattern, since
// this shape updates its accumulator twice per candidate with no early
// exit.
static Value* matchMaxUpdateOtherValue(CallBase* CB, AllocaInst* BestAcc) {
  if (CB->arg_size() < 2 || !demangleContains(CB, "std::max<"))
    return nullptr;
  Value *A = CB->getArgOperand(0), *B = CB->getArgOperand(1);
  Value* OtherArg = nullptr;
  if (stripToContainerSource(A) == BestAcc)
    OtherArg = B;
  else if (stripToContainerSource(B) == BestAcc)
    OtherArg = A;
  else
    return nullptr;

  // Result must be loaded back and stored into BestAcc.
  bool StoresBack = false;
  for (User* U : CB->users()) {
    auto* Ld = dyn_cast<LoadInst>(U);
    if (!Ld)
      continue;
    for (User* U2 : Ld->users())
      if (auto* SI = dyn_cast<StoreInst>(U2))
        if (SI->getPointerOperand() == BestAcc)
          StoresBack = true;
  }
  if (!StoresBack)
    return nullptr;

  // OtherArg is a temporary alloca; find what was stored into it.
  auto* TempAlloca = dyn_cast<AllocaInst>(stripToContainerSource(OtherArg));
  if (!TempAlloca)
    return nullptr;
  for (User* U : TempAlloca->users())
    if (auto* SI = dyn_cast<StoreInst>(U))
      if (SI->getPointerOperand() == TempAlloca)
        return SI->getValueOperand();
  return nullptr;
}

// Phase 1: match maxCliques()'s self-recursive running-max shape.
static std::optional<MaxCliqueMatch> matchMaxCliques(Function& F, LoopInfo& LI) {
  if (F.isDeclaration() || F.arg_size() != 5)
    return std::nullopt;
  if (!F.getReturnType()->isIntegerTy(32))
    return std::nullopt;

  // 1: exactly one self-recursive call.
  SmallVector<CallBase*, 2> SelfCalls;
  for (BasicBlock& BB : F)
    for (Instruction& I : BB)
      if (auto* CB = dyn_cast<CallBase>(&I))
        if (CB->getCalledFunction() == &F)
          SelfCalls.push_back(CB);
  if (SelfCalls.size() != 1)
    return std::nullopt;
  CallBase* SelfCall = SelfCalls[0];
  if (SelfCall->arg_size() != F.arg_size())
    return std::nullopt;

  // 2: the candidate loop enclosing the self-call.
  BasicBlock* SelfCallBB = SelfCall->getParent();
  Loop* CandidateLoop = LI.getLoopFor(SelfCallBB);
  if (!CandidateLoop)
    return std::nullopt;

  // 3: the loop's own induction variable -- header phi with a canonical
  // add-by-1 backedge.
  BasicBlock* LoopHeader = CandidateLoop->getHeader();
  BasicBlock* LoopLatch = CandidateLoop->getLoopLatch();
  BasicBlock* LoopPreheader = CandidateLoop->getLoopPreheader();
  if (!LoopLatch || !LoopPreheader)
    return std::nullopt;
  PHINode* LoopPhi = nullptr;
  for (PHINode& PN : LoopHeader->phis()) {
    if (isAddOne(PN.getIncomingValueForBlock(LoopLatch), &PN)) {
      if (LoopPhi)
        return std::nullopt;
      LoopPhi = &PN;
    }
  }
  if (!LoopPhi)
    return std::nullopt;

  // 4: self-call argument classification -- exactly one argument must be
  // LoopPhi + 1 (StartArg), exactly one other must be FormalArg + 1
  // (SizeArg); everything else threaded through unchanged.
  Argument *StartArg = nullptr, *SizeArg = nullptr;
  unsigned StartIdx = 0, SizeIdx = 0;
  for (unsigned i = 0; i < SelfCall->arg_size(); i++) {
    Value* ArgVal = SelfCall->getArgOperand(i);
    Argument* FormalArg = F.getArg(i);
    if (isAddOne(ArgVal, LoopPhi)) {
      if (StartArg)
        return std::nullopt;
      StartArg = FormalArg;
      StartIdx = i;
    } else if (isAddOne(ArgVal, FormalArg)) {
      if (SizeArg)
        return std::nullopt;
      SizeArg = FormalArg;
      SizeIdx = i;
    }
  }
  if (!StartArg || !SizeArg)
    return std::nullopt;
  for (unsigned i = 0; i < SelfCall->arg_size(); i++) {
    if (i == StartIdx || i == SizeIdx)
      continue;
    if (SelfCall->getArgOperand(i) != F.getArg(i))
      return std::nullopt;
  }

  // 5: the loop must start from StartArg -- confirms the search begins at
  // the function's own `start` parameter, not some unrelated origin.
  if (LoopPhi->getIncomingValueForBlock(LoopPreheader) != StartArg)
    return std::nullopt;

  // 6: NArg -- the loop's bound comparison against a third, distinct
  // argument. performReplacement() emits @clique_impl with a fixed i32 N
  // parameter (matching the bridge's actual C++ signature), so -- same
  // reasoning as kcolor_pass.cpp's wide_bounds fix -- reject rather than
  // build a CallInst whose argument type doesn't match.
  Argument* NArg = nullptr;
  if (auto* ICmp = dyn_cast_or_null<ICmpInst>(getCondBrCondition(LoopHeader))) {
    for (Value* Op : {ICmp->getOperand(0), ICmp->getOperand(1)}) {
      if (auto* Arg = dyn_cast<Argument>(Op)) {
        if (Arg != StartArg && Arg != SizeArg)
          NArg = Arg;
      }
    }
  }
  if (!NArg)
    return std::nullopt;
  if (!NArg->getType()->isIntegerTy(32))
    return std::nullopt;

  // 7: guard call -- SelfCallBB's unique predecessor branches on a call to
  // some other function, taking that branch on the true edge.
  BasicBlock* GuardBB = SelfCallBB->getUniquePredecessor();
  if (!GuardBB)
    return std::nullopt;
  auto* GuardBI = dyn_cast<CondBrInst>(GuardBB->getTerminator());
  if (!GuardBI || GuardBI->getSuccessor(0) != SelfCallBB)
    return std::nullopt;
  auto* GuardCall = dyn_cast<CallBase>(GuardBI->getCondition());
  if (!GuardCall || GuardCall->getCalledFunction() == &F)
    return std::nullopt;
  Function* GuardFn = GuardCall->getCalledFunction();
  if (!GuardFn)
    return std::nullopt;

  // 8: guard call's argument must be SizeArg + 1 -- a derived value
  // (unlike kcolor's guard, which takes NodeArg raw), structurally
  // re-derived here rather than matched by identity with the other
  // size+1 occurrences.
  bool SizePlusOneInGuard = false;
  for (Value* A : GuardCall->args())
    if (isAddOne(A, SizeArg))
      SizePlusOneInGuard = true;
  if (!SizePlusOneInGuard)
    return std::nullopt;

  // 9: AssignStore -- inside GuardBB, before GuardCall (the assignment
  // must be visible to the guard check, which reads it), a store of
  // LoopPhi into Container[SizeArg]. Identifies CliqueArg. Unlike kcolor,
  // there is no corresponding backtrack store anywhere in this shape --
  // clique[size] is simply overwritten by the next iteration, so none is
  // required here.
  CallBase* AssignIndex = nullptr;
  StoreInst* AssignStore = nullptr;
  Argument* CliqueArg = nullptr;
  for (Instruction& I : *GuardBB) {
    if (&I == GuardCall)
      break;
    auto* SI = dyn_cast<StoreInst>(&I);
    if (!SI || SI->getValueOperand() != LoopPhi)
      continue;
    auto* CB = dyn_cast<CallBase>(SI->getPointerOperand());
    if (!CB || CB->arg_size() < 2 || !isOperatorBracket(CB))
      continue;
    if (stripIntCasts(CB->getArgOperand(1)) != SizeArg)
      continue;
    auto* CArg = dyn_cast_or_null<Argument>(
        stripToContainerSource(CB->getArgOperand(0)));
    if (!CArg || CArg->getParent() != &F)
      continue;
    AssignIndex = CB;
    AssignStore = SI;
    CliqueArg = CArg;
    break;
  }
  if (!AssignStore)
    return std::nullopt;

  // 10: GraphArg -- the one remaining formal parameter, sanity-checked as
  // one of GuardCall's arguments.
  Argument* GraphArg = nullptr;
  for (Argument& A : F.args()) {
    if (&A == StartArg || &A == CliqueArg || &A == SizeArg || &A == NArg)
      continue;
    GraphArg = &A;
    break;
  }
  if (!GraphArg)
    return std::nullopt;
  bool GraphUsedInGuard = false;
  for (Value* A : GuardCall->args())
    if (stripToContainerSource(A) == GraphArg)
      GraphUsedInGuard = true;
  if (!GraphUsedInGuard)
    return std::nullopt;

  // 11: running-max accumulator -- exactly two std::max<int> call sites
  // within the candidate loop, both updating the same memory-resident
  // accumulator (BestAcc): one accepting the just-extended clique
  // (SizeArg + 1), one folding in the recursive search's result.
  SmallVector<CallBase*, 4> MaxCandidates;
  for (BasicBlock* BB : CandidateLoop->blocks())
    for (Instruction& I : *BB)
      if (auto* CB = dyn_cast<CallBase>(&I))
        if (demangleContains(CB, "std::max<"))
          MaxCandidates.push_back(CB);
  if (MaxCandidates.size() != 2)
    return std::nullopt;

  AllocaInst* BestAcc = nullptr;
  for (Value* Cand :
       {MaxCandidates[0]->getArgOperand(0), MaxCandidates[0]->getArgOperand(1)}) {
    auto* AI = dyn_cast_or_null<AllocaInst>(stripToContainerSource(Cand));
    if (!AI)
      continue;
    for (Value* Other : {MaxCandidates[1]->getArgOperand(0),
                        MaxCandidates[1]->getArgOperand(1)})
      if (stripToContainerSource(Other) == AI)
        BestAcc = AI;
  }
  if (!BestAcc)
    return std::nullopt;

  CallBase *AcceptMaxCall = nullptr, *ExtendMaxCall = nullptr;
  for (CallBase* CB : MaxCandidates) {
    Value* Other = matchMaxUpdateOtherValue(CB, BestAcc);
    if (!Other)
      return std::nullopt;
    if (isAddOne(Other, SizeArg)) {
      if (AcceptMaxCall)
        return std::nullopt;
      AcceptMaxCall = CB;
    } else if (Other == SelfCall) {
      if (ExtendMaxCall)
        return std::nullopt;
      ExtendMaxCall = CB;
    }
  }
  if (!AcceptMaxCall || !ExtendMaxCall)
    return std::nullopt;

  return MaxCliqueMatch{&F,          StartArg,    CliqueArg,   SizeArg,
                        NArg,        GraphArg,    CandidateLoop, LoopPhi,
                        SelfCall,    GuardCall,   GuardFn,
                        AssignIndex, AssignStore, BestAcc,
                        AcceptMaxCall, ExtendMaxCall};
}

// Gates

// Reject any side-effecting call in maxCliques()'s body beyond the
// recognised structural calls (self-call, guard call, the clique[size]
// index call, both std::max calls) and generic std::vector helpers.
static bool checkSideEffects(const MaxCliqueMatch& M) {
  for (BasicBlock& BB : *M.MaxCliquesFn) {
    for (Instruction& I : BB) {
      auto* CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      if (CB->onlyReadsMemory() || CB->doesNotAccessMemory())
        continue;
      if (CB == M.SelfCall || CB == M.GuardCall || CB == M.AssignIndex ||
          CB == M.AcceptMaxCall || CB == M.ExtendMaxCall)
        continue;
      Function* Callee = CB->getCalledFunction();
      if (Callee) {
        std::string D = normalizeDemangled(demangle(Callee->getName()));
        if (StringRef(D).contains("std::vector"))
          continue;
      }
      LLVM_DEBUG(dbgs() << "[gate] unrecognised side-effecting call in "
                        << M.MaxCliquesFn->getName() << "(): " << I << "\n");
      return false;
    }
  }
  return true;
}

// Reject any side-effecting call in the guard function's body beyond
// std::vector helpers, and reject cross-recursion into maxCliques() itself.
static bool checkGuardSideEffects(const MaxCliqueMatch& M) {
  for (BasicBlock& BB : *M.GuardFn) {
    for (Instruction& I : BB) {
      auto* CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      if (CB->getCalledFunction() == M.MaxCliquesFn) {
        LLVM_DEBUG(dbgs() << "[gate] guard function calls back into "
                          << M.MaxCliquesFn->getName() << "(): " << I << "\n");
        return false;
      }
      if (CB->onlyReadsMemory() || CB->doesNotAccessMemory())
        continue;
      Function* Callee = CB->getCalledFunction();
      if (Callee) {
        std::string D = normalizeDemangled(demangle(Callee->getName()));
        if (StringRef(D).contains("std::vector"))
          continue;
      }
      LLVM_DEBUG(dbgs() << "[gate] unrecognised side-effecting call in "
                        << M.GuardFn->getName() << "(): " << I << "\n");
      return false;
    }
  }
  return true;
}

// Phase 2: find top-level invocations -- calls to maxCliques() where BOTH
// the StartArg-position and SizeArg-position arguments are the literal
// constant 0. Both are required (not just one, unlike kcolor's single
// node==0 check): findMaxClique() calls maxCliques(0, clique, 0, N,
// graph), and requiring both gives a stronger, more specific anchor than
// either alone.
static void findTopLevelCalls(const MaxCliqueMatch& M,
                              SmallVectorImpl<CallBase*>& Out) {
  unsigned StartIdx = M.StartArg->getArgNo();
  unsigned SizeIdx = M.SizeArg->getArgNo();
  for (User* U : M.MaxCliquesFn->users()) {
    auto* CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledOperand() != M.MaxCliquesFn || CB == M.SelfCall)
      continue;
    if (CB->arg_size() != M.MaxCliquesFn->arg_size())
      continue;
    auto* C0 = dyn_cast<ConstantInt>(CB->getArgOperand(StartIdx));
    auto* C1 = dyn_cast<ConstantInt>(CB->getArgOperand(SizeIdx));
    if (C0 && C0->isZero() && C1 && C1->isZero())
      Out.push_back(CB);
  }
}

// Replacement

// Replace a top-level call/invoke of maxCliques() with a call to
// @clique_impl. Signature: i32 @clique_impl(ptr graph, i32 N, ptr clique_out)
static bool performReplacement(const MaxCliqueMatch& M, CallBase* Call,
                               Module& Mod) {
  LLVMContext& Ctx = Mod.getContext();
  PointerType* PtrTy = PointerType::get(Ctx, 0);
  Type* I32Ty = Type::getInt32Ty(Ctx);

  FunctionType* FTy = FunctionType::get(I32Ty, {PtrTy, I32Ty, PtrTy}, false);
  auto* ImplFn =
      cast<Function>(Mod.getOrInsertFunction("clique_impl", FTy).getCallee());
  ImplFn->setDoesNotThrow();

  Value* GraphVal = Call->getArgOperand(M.GraphArg->getArgNo());
  Value* NVal = Call->getArgOperand(M.NArg->getArgNo());
  Value* CliqueVal = Call->getArgOperand(M.CliqueArg->getArgNo());

  IRBuilder<> Builder(Call);
  CallInst* Result =
      Builder.CreateCall(ImplFn, {GraphVal, NVal, CliqueVal}, "clique_result");

  Call->replaceAllUsesWith(Result);

  if (auto* Invoke = dyn_cast<InvokeInst>(Call)) {
    // Same erase-then-append pattern as kcolor_pass.cpp (ReplaceInstWithInst
    // would attempt its own RAUW between the i32 invoke and the void
    // branch, which asserts on the type mismatch regardless of remaining
    // uses -- see kcolor_pass.cpp's performReplacement for the discovery).
    BasicBlock* NormalDest = Invoke->getNormalDest();
    BasicBlock* UnwindDest = Invoke->getUnwindDest();
    BasicBlock* CallBB = Invoke->getParent();
    UnwindDest->removePredecessor(CallBB, /*KeepOneInputPHIs=*/false);
    Invoke->eraseFromParent();
    UncondBrInst::Create(NormalDest, CallBB);
  } else {
    Call->eraseFromParent();
  }

  errs() << "  *** replaced top-level maxCliques() call with call to "
           "@clique_impl\n\n";
  return true;
}

// Reporting
static void printMatch(const MaxCliqueMatch& M) {
  errs() << "\n  *** Max-clique backtracking pattern matched ***\n";
  errs() << "    function    : " << M.MaxCliquesFn->getName() << "\n";
  errs() << "    loop phi    : " << *M.LoopPhi << "\n";
  errs() << "    self-call   : " << *M.SelfCall << "\n";
  errs() << "    guard call  : " << *M.GuardCall << " (" << M.GuardFn->getName()
        << ")\n";
  errs() << "    assign      : " << *M.AssignStore << "\n";
  errs() << "    accept max  : " << *M.AcceptMaxCall << "\n";
  errs() << "    extend max  : " << *M.ExtendMaxCall << "\n";
  errs() << "  -- Arguments --\n";
  errs() << "    start: #" << M.StartArg->getArgNo() << "\n";
  errs() << "    clique: #" << M.CliqueArg->getArgNo() << "\n";
  errs() << "    size : #" << M.SizeArg->getArgNo() << "\n";
  errs() << "    N    : #" << M.NArg->getArgNo() << "\n";
  errs() << "    graph: #" << M.GraphArg->getArgNo() << "\n\n";
}

// Pass
namespace {
struct CliquePass : PassInfoMixin<CliquePass> {
  PreservedAnalyses run(Module& Mod, ModuleAnalysisManager& MAM) {
    auto& FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(Mod).getManager();

    LLVM_DEBUG(dbgs() << "[Clique] scanning module: " << Mod.getName()
                      << "\n");

    SmallVector<MaxCliqueMatch, 2> Matches;
    for (Function& F : Mod) {
      if (F.isDeclaration())
        continue;
      LoopInfo& LI = FAM.getResult<LoopAnalysis>(F);
      auto Match = matchMaxCliques(F, LI);
      if (!Match)
        continue;
      LLVM_DEBUG(dbgs() << "Found candidate maxCliques()-shaped function: "
                        << F.getName() << "\n");
      if (!checkSideEffects(*Match)) {
        errs() << "  [skip] " << F.getName()
               << ": unaccounted side effects in maxCliques body\n";
        continue;
      }
      if (!checkGuardSideEffects(*Match)) {
        errs() << "  [skip] " << F.getName()
               << ": unaccounted side effects in guard function\n";
        continue;
      }
      Matches.push_back(*Match);
    }

    if (Matches.empty()) {
      LLVM_DEBUG(dbgs() << "  no max-clique pattern found.\n");
      return PreservedAnalyses::all();
    }

    bool Changed = false;
    for (auto& M : Matches) {
      printMatch(M);
      SmallVector<CallBase*, 2> TopLevelCalls;
      findTopLevelCalls(M, TopLevelCalls);
      if (TopLevelCalls.empty()) {
        errs() << "  [note] " << M.MaxCliquesFn->getName()
               << " matched but no top-level (start==0, size==0) call sites "
                  "found\n";
        continue;
      }
      for (CallBase* Call : TopLevelCalls)
        if (performReplacement(M, Call, Mod))
          Changed = true;
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
}  // namespace

void registerCliquePass(PassBuilder& PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager& MPM,
         ArrayRef<PassBuilder::PipelineElement>) -> bool {
        if (Name == "clique-pass") {
          MPM.addPass(CliquePass());
          return true;
        }
        return false;
      });
}
