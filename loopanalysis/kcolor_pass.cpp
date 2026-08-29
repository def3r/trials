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

#define DEBUG_TYPE "kcolor-cpp"  // -debug-only=kcolor-cpp

using namespace llvm;

// Data structures
//
// Unlike MaxCut/TSP, the m-coloring decision problem isn't solved by an
// outer loop enumerating an exponential candidate space -- it's solved by
// self-recursion: solve(node, ...) tries each color at `node`, recurses on
// node+1, and backtracks on failure. There is no single "outer loop" to
// replace; the thing to replace is the *top-level invocation* of the
// recursive machinery (a call to solve() with node == 0), wherever it
// occurs -- which may be in a different function than solve() itself. That
// makes this a Module-level match/replace, not a Function-level one.
struct SolveMatch {
  Function* SolveFn;
  Argument* NodeArg;   // recursion-depth parameter
  Argument* ColorArg;  // color assignment container, indexed by NodeArg
  Argument* GraphArg;  // adjacency-matrix container, threaded through unchanged
  Argument* MArg;      // candidate-loop bound (max colors to try)
  Argument* NArg;      // base-case bound (number of nodes)

  CallBase* SelfCall;   // solve(node+1, color, m, N, graph)
  Loop* CandidateLoop;  // the "try each color" loop enclosing SelfCall
  CallBase* GuardCall;  // call to the safety-check helper (e.g. isSafe)
  Function* GuardFn;

  CallBase*
      AssignIndex;  // color[node] operator[] call (assign, before SelfCall)
  StoreInst* AssignStore;
  CallBase*
      BacktrackIndex;  // color[node] operator[] call (backtrack, after failure)
  StoreInst* BacktrackStore;

  ICmpInst* BaseCaseCmp;  // icmp eq NodeArg, NArg
};

// Helpers (mirrors tsp_pass.cpp / maxcut_pass.cpp; duplicated per that
// established convention of not sharing a common header between passes).

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

// Phase 1: match solve()'s self-recursive backtracking shape.
static std::optional<SolveMatch> matchSolve(Function& F, LoopInfo& LI) {
  if (F.isDeclaration() || F.arg_size() != 5)
    return std::nullopt;
  if (!F.getReturnType()->isIntegerTy(1))
    return std::nullopt;

  // 1.1: exactly one self-recursive call.
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

  // 1.2: recursion parameter -- the one self-call argument that is
  // `add(FormalArg_i, 1)`; every other argument must be threaded through
  // unchanged (same value as the corresponding formal parameter).
  Argument* NodeArg = nullptr;
  unsigned NodeIdx = 0;
  for (unsigned i = 0; i < SelfCall->arg_size(); i++) {
    Argument* FormalArg = F.getArg(i);
    auto* BO = dyn_cast<BinaryOperator>(SelfCall->getArgOperand(i));
    if (!BO || BO->getOpcode() != Instruction::Add)
      continue;
    Value *Op0 = BO->getOperand(0), *Op1 = BO->getOperand(1);
    auto* C = dyn_cast<ConstantInt>(Op1);
    if (Op0 == FormalArg && C && C->isOne()) {
      NodeArg = FormalArg;
      NodeIdx = i;
      break;
    }
  }
  if (!NodeArg)
    return std::nullopt;
  for (unsigned i = 0; i < SelfCall->arg_size(); i++) {
    if (i == NodeIdx)
      continue;
    if (SelfCall->getArgOperand(i) != F.getArg(i))
      return std::nullopt;
  }

  // 1.3: the candidate loop enclosing the self-call.
  BasicBlock* SelfCallBB = SelfCall->getParent();
  Loop* CandidateLoop = LI.getLoopFor(SelfCallBB);
  if (!CandidateLoop)
    return std::nullopt;

  // 1.4: guard call -- SelfCallBB's unique predecessor branches on a call
  // to some other function, taking that branch on the true edge.
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
  bool NodeUsedInGuard = false;
  for (Value* A : GuardCall->args())
    if (A == NodeArg)
      NodeUsedInGuard = true;
  if (!NodeUsedInGuard)
    return std::nullopt;

  // 1.5: AssignStore -- inside SelfCallBB, before SelfCall, a store into
  // Container[NodeArg]. Identifies ColorArg.
  CallBase* AssignIndex = nullptr;
  StoreInst* AssignStore = nullptr;
  Argument* ColorArg = nullptr;
  for (Instruction& I : *SelfCallBB) {
    if (&I == SelfCall)
      break;
    auto* SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;
    auto* CB = dyn_cast<CallBase>(SI->getPointerOperand());
    if (!CB || CB->arg_size() < 2 || !isOperatorBracket(CB))
      continue;
    if (stripIntCasts(CB->getArgOperand(1)) != NodeArg)
      continue;
    auto* CArg = dyn_cast_or_null<Argument>(
        stripToContainerSource(CB->getArgOperand(0)));
    if (!CArg || CArg->getParent() != &F || CArg == NodeArg)
      continue;
    AssignIndex = CB;
    AssignStore = SI;
    ColorArg = CArg;
    break;
  }
  if (!AssignStore)
    return std::nullopt;

  // 1.6: base case -- entry block's branch condition is `icmp eq NodeArg,
  // <other argument>`.
  ICmpInst* BaseCaseCmp = nullptr;
  Argument* NArg = nullptr;
  BasicBlock* Entry = &F.getEntryBlock();
  if (auto* ICmp = dyn_cast_or_null<ICmpInst>(getCondBrCondition(Entry))) {
    if (ICmp->getPredicate() == ICmpInst::ICMP_EQ) {
      Value *Op0 = ICmp->getOperand(0), *Op1 = ICmp->getOperand(1);
      if (Op0 == NodeArg)
        NArg = dyn_cast<Argument>(Op1);
      else if (Op1 == NodeArg)
        NArg = dyn_cast<Argument>(Op0);
      if (NArg)
        BaseCaseCmp = ICmp;
    }
  }
  if (!BaseCaseCmp)
    return std::nullopt;
  // performReplacement() emits @kcolor_impl with a fixed i32/i32 signature
  // for m/N (matching the bridge function's actual C++ signature) -- a
  // wider or narrower NArg would build a CallInst whose argument type
  // doesn't match that declared parameter type, which is not a missed
  // match but a verifier-rejected/crash-prone construct. Reject here
  // instead of building bad IR.
  if (!NArg->getType()->isIntegerTy(32))
    return std::nullopt;

  // 1.7: candidate loop's bound -- the header's branch condition compares
  // the loop phi against some other argument (distinct from what's found
  // so far). That argument is MArg.
  Argument* MArg = nullptr;
  if (auto* ICmp = dyn_cast_or_null<ICmpInst>(
          getCondBrCondition(CandidateLoop->getHeader()))) {
    for (Value* Op : {ICmp->getOperand(0), ICmp->getOperand(1)}) {
      if (auto* Arg = dyn_cast<Argument>(Op)) {
        if (Arg != NodeArg && Arg != ColorArg && Arg != NArg)
          MArg = Arg;
      }
    }
  }
  if (!MArg)
    return std::nullopt;
  if (!MArg->getType()->isIntegerTy(32))
    return std::nullopt;  // same reasoning as the NArg check above

  // 1.8: GraphArg -- the one remaining formal parameter (F has exactly 5:
  // NodeArg, ColorArg, NArg, MArg, and this one), sanity-checked as one of
  // GuardCall's arguments.
  Argument* GraphArg = nullptr;
  for (Argument& A : F.args()) {
    if (&A == NodeArg || &A == ColorArg || &A == NArg || &A == MArg)
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

  // 1.9: BacktrackStore -- on the false edge of SelfCall's boolean result,
  // another store into Container[NodeArg] (same container as AssignStore).
  CallBase* BacktrackIndex = nullptr;
  StoreInst* BacktrackStore = nullptr;
  for (User* U : SelfCall->users()) {
    auto* CBR = dyn_cast<CondBrInst>(U);
    if (!CBR || CBR->getCondition() != SelfCall)
      continue;
    BasicBlock* FalseBB = CBR->getSuccessor(1);
    for (Instruction& I : *FalseBB) {
      auto* SI = dyn_cast<StoreInst>(&I);
      if (!SI)
        continue;
      auto* CB = dyn_cast<CallBase>(SI->getPointerOperand());
      if (!CB || CB->arg_size() < 2 || !isOperatorBracket(CB))
        continue;
      if (stripIntCasts(CB->getArgOperand(1)) != NodeArg)
        continue;
      if (stripToContainerSource(CB->getArgOperand(0)) != ColorArg)
        continue;
      BacktrackIndex = CB;
      BacktrackStore = SI;
      break;
    }
    break;
  }
  if (!BacktrackStore)
    return std::nullopt;

  return SolveMatch{
      &F,          NodeArg,     ColorArg,       GraphArg,       MArg,
      NArg,        SelfCall,    CandidateLoop,  GuardCall,      GuardFn,
      AssignIndex, AssignStore, BacktrackIndex, BacktrackStore, BaseCaseCmp};
}

// Gates

// Reject any side-effecting call in solve()'s body beyond the recognised
// structural calls (self-call, guard call, the two color[node] index
// calls) and generic std::vector helpers.
static bool checkSolveSideEffects(const SolveMatch& M) {
  for (BasicBlock& BB : *M.SolveFn) {
    for (Instruction& I : BB) {
      auto* CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      if (CB->onlyReadsMemory() || CB->doesNotAccessMemory())
        continue;
      if (CB == M.SelfCall || CB == M.GuardCall || CB == M.AssignIndex ||
          CB == M.BacktrackIndex)
        continue;
      Function* Callee = CB->getCalledFunction();
      if (Callee) {
        std::string D = normalizeDemangled(demangle(Callee->getName()));
        if (StringRef(D).contains("std::vector"))
          continue;
      }
      LLVM_DEBUG(dbgs() << "[gate] unrecognised side-effecting call in "
                        << M.SolveFn->getName() << "(): " << I << "\n");
      return false;
    }
  }
  return true;
}

// Reject any side-effecting call in the guard function's body beyond
// std::vector helpers, and reject cross-recursion into solve() itself.
static bool checkGuardSideEffects(const SolveMatch& M) {
  for (BasicBlock& BB : *M.GuardFn) {
    for (Instruction& I : BB) {
      auto* CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      if (CB->getCalledFunction() == M.SolveFn) {
        LLVM_DEBUG(dbgs() << "[gate] guard function calls back into solve(): "
                          << I << "\n");
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

// Phase 2: find top-level invocations -- calls to solve() where the
// NodeArg-position argument is the literal constant 0. Recursion only ever
// advances via node+1, so a literal 0 is a sound signal of "start a fresh
// coloring attempt" rather than a mid-recursion value.
static void findTopLevelCalls(const SolveMatch& M,
                              SmallVectorImpl<CallBase*>& Out) {
  unsigned NodeIdx = M.NodeArg->getArgNo();
  for (User* U : M.SolveFn->users()) {
    auto* CB = dyn_cast<CallBase>(U);
    if (!CB || CB->getCalledOperand() != M.SolveFn || CB == M.SelfCall)
      continue;
    if (CB->arg_size() != M.SolveFn->arg_size())
      continue;
    auto* C = dyn_cast<ConstantInt>(CB->getArgOperand(NodeIdx));
    if (C && C->isZero())
      Out.push_back(CB);
  }
}

// Replacement

// Replace a top-level call/invoke of solve() with a call to @kcolor_impl.
// Signature: i1 @kcolor_impl(ptr graph, i32 m, i32 N, ptr color_out)
static bool performReplacement(const SolveMatch& M,
                               CallBase* Call,
                               Module& Mod) {
  LLVMContext& Ctx = Mod.getContext();
  PointerType* PtrTy = PointerType::get(Ctx, 0);
  Type* I32Ty = Type::getInt32Ty(Ctx);

  FunctionType* FTy = FunctionType::get(Type::getInt1Ty(Ctx),
                                        {PtrTy, I32Ty, I32Ty, PtrTy}, false);
  auto* ImplFn =
      cast<Function>(Mod.getOrInsertFunction("kcolor_impl", FTy).getCallee());
  ImplFn->setDoesNotThrow();

  Value* GraphVal = Call->getArgOperand(M.GraphArg->getArgNo());
  Value* MVal = Call->getArgOperand(M.MArg->getArgNo());
  Value* NVal = Call->getArgOperand(M.NArg->getArgNo());
  Value* ColorVal = Call->getArgOperand(M.ColorArg->getArgNo());

  IRBuilder<> Builder(Call);
  CallInst* Result = Builder.CreateCall(
      ImplFn, {GraphVal, MVal, NVal, ColorVal}, "kcolor_result");

  Call->replaceAllUsesWith(Result);

  if (auto* Invoke = dyn_cast<InvokeInst>(Call)) {
    // Invoke->replaceAllUsesWith(Result) already happened above, so it's
    // safe to erase; ReplaceInstWithInst would be wrong here regardless --
    // it does its own unconditional RAUW(OldTerminator, NewTerminator),
    // which asserts on the i1-vs-void type mismatch even with zero uses
    // left. Mirror the erase-then-append pattern tsp_pass.cpp/
    // maxcut_pass.cpp already use for preheader redirection instead.
    BasicBlock* NormalDest = Invoke->getNormalDest();
    BasicBlock* UnwindDest = Invoke->getUnwindDest();
    BasicBlock* CallBB = Invoke->getParent();
    UnwindDest->removePredecessor(CallBB, /*KeepOneInputPHIs=*/false);
    Invoke->eraseFromParent();
    UncondBrInst::Create(NormalDest, CallBB);
  } else {
    Call->eraseFromParent();
  }

  errs() << "  *** replaced top-level solve() call with call to "
            "@kcolor_impl\n\n";
  return true;
}

// Reporting
static void printMatch(const SolveMatch& M) {
  errs() << "\n  *** KColor backtracking pattern matched ***\n";
  errs() << "    function    : " << M.SolveFn->getName() << "\n";
  errs() << "    self-call   : " << *M.SelfCall << "\n";
  errs() << "    base case   : " << *M.BaseCaseCmp << "\n";
  errs() << "    guard call  : " << *M.GuardCall << " (" << M.GuardFn->getName()
         << ")\n";
  errs() << "    assign      : " << *M.AssignStore << "\n";
  errs() << "    backtrack   : " << *M.BacktrackStore << "\n";
  errs() << "  -- Arguments --\n";
  errs() << "    node : #" << M.NodeArg->getArgNo() << "\n";
  errs() << "    color: #" << M.ColorArg->getArgNo() << "\n";
  errs() << "    m    : #" << M.MArg->getArgNo() << "\n";
  errs() << "    N    : #" << M.NArg->getArgNo() << "\n";
  errs() << "    graph: #" << M.GraphArg->getArgNo() << "\n\n";
}

// Pass
namespace {
struct KColorPass : PassInfoMixin<KColorPass> {
  PreservedAnalyses run(Module& Mod, ModuleAnalysisManager& MAM) {
    auto& FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(Mod).getManager();

    LLVM_DEBUG(dbgs() << "[KColor] scanning module: " << Mod.getName() << "\n");

    SmallVector<SolveMatch, 2> Matches;
    for (Function& F : Mod) {
      if (F.isDeclaration())
        continue;
      LoopInfo& LI = FAM.getResult<LoopAnalysis>(F);
      auto Match = matchSolve(F, LI);
      if (!Match)
        continue;
      LLVM_DEBUG(dbgs() << "Found candidate solve()-shaped function: "
                        << F.getName() << "\n");
      if (!checkSolveSideEffects(*Match)) {
        errs() << "  [skip] " << F.getName()
               << ": unaccounted side effects in solve body\n";
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
      LLVM_DEBUG(dbgs() << "  no KColor pattern found.\n");
      return PreservedAnalyses::all();
    }

    bool Changed = false;
    for (auto& M : Matches) {
      printMatch(M);
      SmallVector<CallBase*, 2> TopLevelCalls;
      findTopLevelCalls(M, TopLevelCalls);
      if (TopLevelCalls.empty()) {
        errs() << "  [note] " << M.SolveFn->getName()
               << " matched but no top-level (node==0) call sites found\n";
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

void registerKColorPass(PassBuilder& PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager& MPM,
         ArrayRef<PassBuilder::PipelineElement>) -> bool {
        if (Name == "kcolor-pass") {
          MPM.addPass(KColorPass());
          return true;
        }
        return false;
      });
}
