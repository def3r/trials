#include <optional>
#include <string>
#include "llvm/ADT/SmallPtrSet.h"
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

#define DEBUG_TYPE "tsp-cpp"  // -debug-only=tsp-cpp

using namespace llvm;

// Data structures
//
// TSP (permutation branch-and-bound-free / brute-force form) nests the same
// way MaxCut does: an inner "scoring" loop walks a candidate permutation and
// accumulates a path cost, inside an outer loop that walks candidates
// (successive permutations) and tracks the best (minimum) cost seen.
//
// Unlike MaxCut's accumulator/max-tracker, which SROA/mem2reg promotes to
// SSA phis, TSP's cost/min accumulators have their address taken (passed by
// reference into std::min), so they remain memory-resident (alloca +
// load/store) even after optimization. The matchers below work on that
// load-add-store shape instead of phi backedges.
struct TspScoringMatch {
  Loop* L;
  PHINode* IdxPhi;          // trip index into the permutation array
  PHINode* PrevNodePhi;     // carried "current city", preheader init = const
  AllocaInst* CostAcc;      // currCost-equivalent alloca, preheader init = 0
  BinaryOperator* CostAdd;  // add (load CostAcc), (load cost[cur][next])
  CallBase* Index1;         // cost[currNode]            (row lookup)
  CallBase* Index2;         // cost[currNode][nodes[i]]  (element lookup)
  CallBase* PermIndex;      // nodes[i]  (permutation-array element access)
  Value* PermContainer;     // "nodes" alloca/arg
  Value* CostMatrix;        // "cost" alloca/arg
};

struct TspMatch {
  TspScoringMatch Inner;

  Loop* OuterL;
  AllocaInst* MinAcc;      // minCost-equivalent alloca, preheader init = const
  CallBase* NextPermCall;  // std::next_permutation(begin, end); i1 result
                           // drives the outer loop's back-edge condition

  // Mandatory "wrap to start" epilogue: currCost += cost[currNode][0],
  // executed once per outer iteration after the inner loop exits.
  CallBase* CloseIndex1;
  CallBase* CloseIndex2;
  BinaryOperator* CloseAdd;

  // Min-update: exactly one of these is set.
  CallBase* MinCallForm;  // call to std::min<...>(ptr MinAcc, ptr CostAcc)
  ICmpInst* MinCmpForm;   // icmp slt/sgt (load CostAcc, load MinAcc)
};

// Helpers

// Trace through call chains and casts to reach an AllocaInst or Argument.
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

// Strip a chain of sext/zext/trunc casts (e.g. the `sext i32 %x to i64`
// clang emits around signed array indices).
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

// Return the unique non-EH exit block of L, or nullptr.
static BasicBlock* getNormalExitBlock(Loop* L) {
  BasicBlock* ExitBB = nullptr;
  for (BasicBlock* BB : L->blocks()) {
    for (BasicBlock* Succ : successors(BB)) {
      if (L->contains(Succ) || Succ->isEHPad())
        continue;
      if (ExitBB && ExitBB != Succ)
        return nullptr;
      ExitBB = Succ;
    }
  }
  return ExitBB;
}

static Value* getCondBrCondition(BasicBlock* BB) {
  if (auto* CBR = dyn_cast<CondBrInst>(BB->getTerminator()))
    return CBR->getCondition();
  return nullptr;
}

static void collectAllLoops(Loop* Root, SmallVectorImpl<Loop*>& Out) {
  Out.push_back(Root);
  for (Loop* Sub : Root->getSubLoops())
    collectAllLoops(Sub, Out);
}

// A `container[index]` access lowered as a call to some `operator[]`.
static bool isOperatorBracket(const CallBase* CB) {
  return demangleContains(CB, "operator[]");
}

// Trace `V` (after unwrapping the ashr-exact(shl X,32),32 idiom instcombine
// emits for `sext(trunc(X))`, plus any plain int casts) back to a call to
// `Container.size()`.
static CallBase* traceSizeCall(Value* V, Value* Container) {
  if (auto* Ashr = dyn_cast<BinaryOperator>(V)) {
    if (Ashr->getOpcode() == Instruction::AShr) {
      if (auto* Shl = dyn_cast<BinaryOperator>(Ashr->getOperand(0))) {
        if (Shl->getOpcode() == Instruction::Shl)
          V = Shl->getOperand(0);
      }
    }
  }
  V = stripIntCasts(V);
  auto* CB = dyn_cast<CallBase>(V);
  if (!CB || CB->arg_size() < 1 || !demangleContains(CB, "size()"))
    return nullptr;
  if (stripToContainerSource(CB->getArgOperand(0)) != Container)
    return nullptr;
  return CB;
}

// Match `Store(Add(Load(AccAlloca), Load(Index2)), AccAlloca)` where
// Index2 = Cont2[SecondIdx], Index2's `this` = Index1 = Cont1[FirstIdx].
// Returns {Index1, Index2, CostAdd} on success.
struct DoubleIndexAdd {
  CallBase* Index1;
  CallBase* Index2;
  BinaryOperator* Add;
};

static std::optional<DoubleIndexAdd> matchDoubleIndexAdd(
    BasicBlock* BB,
    AllocaInst* AccAlloca,
    Value* FirstIdxExpected,
    Value* Cont1Expected,
    llvm::function_ref<bool(Value*)> matchesSecondIdx) {
  for (Instruction& I : *BB) {
    auto* SI = dyn_cast<StoreInst>(&I);
    if (!SI || SI->getPointerOperand() != AccAlloca)
      continue;
    auto* Add = dyn_cast<BinaryOperator>(SI->getValueOperand());
    if (!Add || Add->getOpcode() != Instruction::Add)
      continue;
    auto isSelfLoad = [&](Value* V) {
      auto* L = dyn_cast<LoadInst>(V);
      return L && L->getPointerOperand() == AccAlloca;
    };
    Value *Op0 = Add->getOperand(0), *Op1 = Add->getOperand(1);
    Value* Other = isSelfLoad(Op0) ? Op1 : (isSelfLoad(Op1) ? Op0 : nullptr);
    if (!Other)
      continue;
    auto* Ld = dyn_cast<LoadInst>(Other);
    if (!Ld)
      continue;
    auto* Index2 = dyn_cast<CallBase>(Ld->getPointerOperand());
    if (!Index2 || Index2->arg_size() < 2 || !isOperatorBracket(Index2))
      continue;
    if (!matchesSecondIdx(stripIntCasts(Index2->getArgOperand(1))))
      continue;
    auto* Index1 = dyn_cast<CallBase>(Index2->getArgOperand(0));
    if (!Index1 || Index1->arg_size() < 2 || !isOperatorBracket(Index1))
      continue;
    if (stripToContainerSource(Index1->getArgOperand(0)) != Cont1Expected)
      continue;
    if (stripIntCasts(Index1->getArgOperand(1)) != FirstIdxExpected)
      continue;
    return DoubleIndexAdd{Index1, Index2, Add};
  }
  return std::nullopt;
}

// Phase 1: match the inner scoring loop.
static std::optional<TspScoringMatch> matchTspScoringLoop(Loop* L) {
  BasicBlock* Header = L->getHeader();
  BasicBlock* Preheader = L->getLoopPreheader();
  BasicBlock* Latch = L->getLoopLatch();
  if (!Preheader || !Latch)
    return std::nullopt;

  // 1.1: Exactly 2 header phis: IdxPhi (backedge = add-by-1) and
  // PrevNodePhi (everything else).
  PHINode *IdxPhi = nullptr, *PrevNodePhi = nullptr;
  unsigned PhiCount = 0;
  for (PHINode& PN : Header->phis()) {
    if (++PhiCount > 2)
      return std::nullopt;
    Value* BackedgeVal = PN.getIncomingValueForBlock(Latch);
    bool IsIdxCandidate = false;
    if (auto* BinOp = dyn_cast<BinaryOperator>(BackedgeVal)) {
      if (BinOp->getOpcode() == Instruction::Add &&
          BinOp->getOperand(0) == &PN) {
        if (auto* C = dyn_cast<ConstantInt>(BinOp->getOperand(1)))
          IsIdxCandidate = C->isOne();
      }
    }
    if (IsIdxCandidate) {
      if (IdxPhi)
        return std::nullopt;
      IdxPhi = &PN;
    } else {
      if (PrevNodePhi)
        return std::nullopt;
      PrevNodePhi = &PN;
    }
  }
  if (!IdxPhi || !PrevNodePhi)
    return std::nullopt;

  auto* IdxInit =
      dyn_cast<ConstantInt>(IdxPhi->getIncomingValueForBlock(Preheader));
  if (!IdxInit || !IdxInit->isZero())
    return std::nullopt;
  if (!isa<ConstantInt>(PrevNodePhi->getIncomingValueForBlock(Preheader)))
    return std::nullopt;

  // 1.2: Loop condition: icmp slt IdxPhi, <size-of-permutation-container>.
  auto* ICmp = dyn_cast_or_null<ICmpInst>(getCondBrCondition(Header));
  if (!ICmp || ICmp->getPredicate() != ICmpInst::ICMP_SLT)
    return std::nullopt;
  if (ICmp->getOperand(0) != IdxPhi)
    return std::nullopt;
  auto* BI = cast<CondBrInst>(Header->getTerminator());
  if (!L->contains(BI->getSuccessor(0)) || L->contains(BI->getSuccessor(1)))
    return std::nullopt;
  Value* End = ICmp->getOperand(1);

  // Build subloop block set so we can skip them in inner scans.
  SmallPtrSet<BasicBlock*, 16> SubLoopBlocks;
  for (Loop* Sub : L->getSubLoops())
    for (BasicBlock* BB : Sub->blocks())
      SubLoopBlocks.insert(BB);

  // 1.3: Locate `currCost += cost[currNode][nodes[i]]` (CostAdd).
  AllocaInst* CostAcc = nullptr;
  BinaryOperator* CostAdd = nullptr;
  CallBase *Index1 = nullptr, *Index2 = nullptr, *PermIndexA = nullptr;
  Value* PermContainer = nullptr;
  Value* CostMatrix = nullptr;

  for (BasicBlock* BB : L->blocks()) {
    if (BB == Header || SubLoopBlocks.count(BB))
      continue;
    for (Instruction& I : *BB) {
      auto* SI = dyn_cast<StoreInst>(&I);
      if (!SI)
        continue;
      auto* AI = dyn_cast<AllocaInst>(SI->getPointerOperand());
      if (!AI)
        continue;
      auto* Add = dyn_cast<BinaryOperator>(SI->getValueOperand());
      if (!Add || Add->getOpcode() != Instruction::Add)
        continue;
      auto isSelfLoad = [&](Value* V) {
        auto* Ld = dyn_cast<LoadInst>(V);
        return Ld && Ld->getPointerOperand() == AI;
      };
      Value *Op0 = Add->getOperand(0), *Op1 = Add->getOperand(1);
      Value* IncrLoadVal =
          isSelfLoad(Op0) ? Op1 : (isSelfLoad(Op1) ? Op0 : nullptr);
      if (!IncrLoadVal)
        continue;
      auto* IncrLoad = dyn_cast<LoadInst>(IncrLoadVal);
      if (!IncrLoad)
        continue;
      auto* Idx2 = dyn_cast<CallBase>(IncrLoad->getPointerOperand());
      if (!Idx2 || Idx2->arg_size() < 2 || !isOperatorBracket(Idx2))
        continue;
      auto* PermLoad =
          dyn_cast<LoadInst>(stripIntCasts(Idx2->getArgOperand(1)));
      if (!PermLoad)
        continue;
      auto* PermIdx = dyn_cast<CallBase>(PermLoad->getPointerOperand());
      if (!PermIdx || PermIdx->arg_size() < 2 || !isOperatorBracket(PermIdx))
        continue;
      if (stripIntCasts(PermIdx->getArgOperand(1)) != IdxPhi)
        continue;
      Value* PC = stripToContainerSource(PermIdx->getArgOperand(0));
      if (!PC)
        continue;
      auto* Idx1 = dyn_cast<CallBase>(Idx2->getArgOperand(0));
      if (!Idx1 || Idx1->arg_size() < 2 || !isOperatorBracket(Idx1))
        continue;
      if (stripIntCasts(Idx1->getArgOperand(1)) != PrevNodePhi)
        continue;
      Value* CM = stripToContainerSource(Idx1->getArgOperand(0));
      if (!CM)
        continue;

      CostAcc = AI;
      CostAdd = Add;
      Index2 = Idx2;
      Index1 = Idx1;
      PermIndexA = PermIdx;
      PermContainer = PC;
      CostMatrix = CM;
      break;
    }
    if (CostAdd)
      break;
  }
  if (!CostAdd)
    return std::nullopt;

  // 1.4: PrevNodePhi's backedge value = load(PermContainer[IdxPhi]).
  auto* PrevBackLoad =
      dyn_cast<LoadInst>(PrevNodePhi->getIncomingValueForBlock(Latch));
  if (!PrevBackLoad)
    return std::nullopt;
  auto* PermIndexB = dyn_cast<CallBase>(PrevBackLoad->getPointerOperand());
  if (!PermIndexB || PermIndexB->arg_size() < 2 ||
      !isOperatorBracket(PermIndexB))
    return std::nullopt;
  if (stripToContainerSource(PermIndexB->getArgOperand(0)) != PermContainer)
    return std::nullopt;
  if (stripIntCasts(PermIndexB->getArgOperand(1)) != IdxPhi)
    return std::nullopt;

  // 1.5: Loop bound End == PermContainer.size().
  CallBase* SizeCall = traceSizeCall(End, PermContainer);
  if (!SizeCall)
    return std::nullopt;

  // 1.6: Preheader zero-inits CostAcc.
  bool FoundZeroInit = false;
  for (Instruction& I : *Preheader) {
    auto* SI = dyn_cast<StoreInst>(&I);
    if (SI && SI->getPointerOperand() == CostAcc) {
      auto* C = dyn_cast<ConstantInt>(SI->getValueOperand());
      if (C && C->isZero())
        FoundZeroInit = true;
    }
  }
  if (!FoundZeroInit)
    return std::nullopt;

  // 1.7: Inner-loop side-effect check — only the recognised operator[]
  // calls may have side effects; everything else in the loop must be
  // read-only (loads, index arithmetic, casts).
  for (BasicBlock* BB : L->blocks()) {
    if (BB == Header || SubLoopBlocks.count(BB))
      continue;
    for (Instruction& I : *BB) {
      auto* CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      if (CB->onlyReadsMemory() || CB->doesNotAccessMemory())
        continue;
      if (CB == Index1 || CB == Index2 || CB == PermIndexA || CB == PermIndexB)
        continue;
      LLVM_DEBUG(dbgs() << "[inner-check] unrecognised side-effecting call "
                           "in scoring loop: "
                        << I << "\n");
      return std::nullopt;
    }
  }

  return TspScoringMatch{
      L,      IdxPhi, PrevNodePhi, CostAcc,       CostAdd,
      Index1, Index2, PermIndexA,  PermContainer, CostMatrix};
}

// Phase 2: match the enclosing permutation loop.
static std::optional<TspMatch> matchTsp(const TspScoringMatch& Inner) {
  Loop* OuterL = Inner.L->getParentLoop();
  if (!OuterL)
    return std::nullopt;

  BasicBlock* OuterHeader = OuterL->getHeader();
  BasicBlock* OuterPreheader = OuterL->getLoopPreheader();
  BasicBlock* OuterLatch = OuterL->getLoopLatch();
  if (!OuterPreheader || !OuterLatch)
    return std::nullopt;

  // 2.1: The inner loop's preheader must be the outer loop's header (the
  // do-while body falls straight into the scoring loop after resetting
  // CostAcc to 0).
  if (Inner.L->getLoopPreheader() != OuterHeader)
    return std::nullopt;

  // 2.2: MinAcc — a distinct alloca const-initialised in the outer preheader.
  AllocaInst* MinAcc = nullptr;
  for (Instruction& I : *OuterPreheader) {
    auto* SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;
    auto* AI = dyn_cast<AllocaInst>(SI->getPointerOperand());
    if (!AI || AI == Inner.CostAcc)
      continue;
    if (!isa<ConstantInt>(SI->getValueOperand()))
      continue;
    MinAcc = AI;
    break;
  }
  if (!MinAcc)
    return std::nullopt;

  // 2.3: NextPermCall drives the outer loop's back-edge condition.
  auto* NextPermCall =
      dyn_cast_or_null<CallBase>(getCondBrCondition(OuterLatch));
  if (!NextPermCall ||
      !demangleContains(NextPermCall, "std::next_permutation<"))
    return std::nullopt;
  auto* OuterBI = cast<CondBrInst>(OuterLatch->getTerminator());
  if (!OuterL->contains(OuterBI->getSuccessor(0)) ||
      OuterL->contains(OuterBI->getSuccessor(1)))
    return std::nullopt;
  if (NextPermCall->arg_size() < 2)
    return std::nullopt;
  auto traceBeginEnd = [](Value* V) -> Value* {
    auto* CB = dyn_cast<CallBase>(V);
    if (!CB || CB->arg_size() < 1)
      return nullptr;
    return stripToContainerSource(CB->getArgOperand(0));
  };
  Value* C0 = traceBeginEnd(NextPermCall->getArgOperand(0));
  Value* C1 = traceBeginEnd(NextPermCall->getArgOperand(1));
  if (!C0 || C0 != C1 || C0 != Inner.PermContainer)
    return std::nullopt;

  SmallPtrSet<BasicBlock*, 8> InnerBlocks;
  for (BasicBlock* BB : Inner.L->blocks())
    InnerBlocks.insert(BB);

  // 2.4: Mandatory wrap-to-start epilogue: currCost += cost[currNode][0].
  CallBase *CloseIndex1 = nullptr, *CloseIndex2 = nullptr;
  BinaryOperator* CloseAdd = nullptr;
  for (BasicBlock* BB : OuterL->blocks()) {
    if (BB == OuterHeader || InnerBlocks.count(BB))
      continue;
    auto Match = matchDoubleIndexAdd(BB, Inner.CostAcc, Inner.PrevNodePhi,
                                     Inner.CostMatrix, [](Value* SecondIdx) {
                                       auto* C =
                                           dyn_cast<ConstantInt>(SecondIdx);
                                       return C && C->isZero();
                                     });
    if (Match) {
      CloseIndex1 = Match->Index1;
      CloseIndex2 = Match->Index2;
      CloseAdd = Match->Add;
      break;
    }
  }
  if (!CloseAdd)
    return std::nullopt;

  // 2.5: Min-update — call-based form (std::min<...>(&MinAcc, &CostAcc))
  // preferred; inline-compare form as a fallback.
  CallBase* MinCallForm = nullptr;
  ICmpInst* MinCmpForm = nullptr;

  for (BasicBlock* BB : OuterL->blocks()) {
    if (MinCallForm)
      break;
    if (BB == OuterHeader || InnerBlocks.count(BB))
      continue;
    for (Instruction& I : *BB) {
      auto* CB = dyn_cast<CallBase>(&I);
      if (!CB || CB->arg_size() < 2 || !demangleContains(CB, "std::min<"))
        continue;
      Value* A = stripToContainerSource(CB->getArgOperand(0));
      Value* B = stripToContainerSource(CB->getArgOperand(1));
      Value* Other = nullptr;
      if (A == Inner.CostAcc)
        Other = B;
      else if (B == Inner.CostAcc)
        Other = A;
      if (Other != MinAcc)
        continue;
      bool StoresBack = false;
      for (User* U : CB->users()) {
        auto* Ld = dyn_cast<LoadInst>(U);
        if (!Ld)
          continue;
        for (User* U2 : Ld->users())
          if (auto* SI = dyn_cast<StoreInst>(U2))
            if (SI->getPointerOperand() == MinAcc)
              StoresBack = true;
      }
      if (!StoresBack)
        continue;
      MinCallForm = CB;
      break;
    }
  }

  if (!MinCallForm) {
    auto loadOf = [](Value* V, AllocaInst* AI) {
      auto* Ld = dyn_cast<LoadInst>(V);
      return Ld && Ld->getPointerOperand() == AI;
    };
    for (BasicBlock* BB : OuterL->blocks()) {
      if (MinCmpForm)
        break;
      if (BB == OuterHeader || InnerBlocks.count(BB))
        continue;
      for (Instruction& I : *BB) {
        auto* Cmp = dyn_cast<ICmpInst>(&I);
        if (!Cmp)
          continue;
        bool CostFirst = Cmp->getPredicate() == ICmpInst::ICMP_SLT;
        bool CostSecond = Cmp->getPredicate() == ICmpInst::ICMP_SGT;
        if (!CostFirst && !CostSecond)
          continue;
        bool Ok = CostFirst ? (loadOf(Cmp->getOperand(0), Inner.CostAcc) &&
                               loadOf(Cmp->getOperand(1), MinAcc))
                            : (loadOf(Cmp->getOperand(0), MinAcc) &&
                               loadOf(Cmp->getOperand(1), Inner.CostAcc));
        if (!Ok)
          continue;
        // Select form: select(Cmp, load CostAcc, load MinAcc) -> store MinAcc.
        for (User* U : Cmp->users()) {
          auto* Sel = dyn_cast<SelectInst>(U);
          if (!Sel)
            continue;
          for (User* U2 : Sel->users())
            if (auto* SI = dyn_cast<StoreInst>(U2))
              if (SI->getPointerOperand() == MinAcc)
                MinCmpForm = Cmp;
          if (MinCmpForm)
            break;
        }
        if (MinCmpForm)
          break;
        // Branching form: true-successor stores load(CostAcc) into MinAcc.
        if (auto* CBR = dyn_cast<CondBrInst>(BB->getTerminator())) {
          if (CBR->getCondition() == Cmp) {
            for (Instruction& TI : *CBR->getSuccessor(0)) {
              auto* SI = dyn_cast<StoreInst>(&TI);
              if (SI && SI->getPointerOperand() == MinAcc &&
                  loadOf(SI->getValueOperand(), Inner.CostAcc)) {
                MinCmpForm = Cmp;
                break;
              }
            }
          }
        }
        if (MinCmpForm)
          break;
      }
    }
  }
  if (!MinCallForm && !MinCmpForm)
    return std::nullopt;

  return TspMatch{Inner,       OuterL,   MinAcc,      NextPermCall, CloseIndex1,
                  CloseIndex2, CloseAdd, MinCallForm, MinCmpForm};
}

// Gates

// Gate 1: the only value that may escape the outer loop is MinAcc's final
// value; since MinAcc is memory-resident (not a phi), that's equivalent to
// requiring the normal exit block to carry no LCSSA phis at all.
static bool checkLiveOuts(const TspMatch& M) {
  BasicBlock* ExitBB = getNormalExitBlock(M.OuterL);
  if (!ExitBB)
    return false;
  for (PHINode& PN : ExitBB->phis()) {
    (void)PN;
    LLVM_DEBUG(dbgs() << "[gate1] unexpected live-out phi: " << PN << "\n");
    return false;
  }
  return true;
}

// Gate 2: scan outer-loop-non-inner blocks for side-effecting calls; only
// container operator[]/begin/end, std::min, and std::next_permutation calls
// are allowed.
//
// A std::vector call whose `this` pointer traces to something other than
// the permutation container itself (Inner.PermContainer) needs extra care:
// an assignment/copy call in that shape (`bestPath = nodes;` on
// improvement, mirroring MaxCut's best_S pattern) writes the winning
// permutation into a separate output container. Unlike MaxCut,
// performReplacement()/tsp_impl have no mechanism to preserve that side
// effect — silently allowing it here would make the pass replace the loop
// and drop the assignment. Reject rather than silently replace.
static bool checkSideEffects(const TspMatch& M) {
  SmallPtrSet<BasicBlock*, 8> InnerBlocks;
  for (BasicBlock* BB : M.Inner.L->blocks())
    InnerBlocks.insert(BB);

  for (BasicBlock* BB : M.OuterL->blocks()) {
    if (InnerBlocks.count(BB))
      continue;
    for (Instruction& I : *BB) {
      auto* CB = dyn_cast<CallBase>(&I);
      if (!CB)
        continue;
      if (CB->onlyReadsMemory() || CB->doesNotAccessMemory())
        continue;
      Function* Callee = CB->getCalledFunction();
      if (!Callee) {
        LLVM_DEBUG(dbgs() << "[gate2] indirect call: " << I << "\n");
        return false;
      }
      std::string D = normalizeDemangled(demangle(Callee->getName()));
      StringRef DS(D);
      if (DS.contains("std::min<") || DS.contains("std::next_permutation<"))
        continue;
      if (DS.contains("std::vector")) {
        bool IsAssignLike = DS.contains("operator=") ||
                            DS.contains("_M_assign") || DS.contains("vector(");
        if (IsAssignLike && CB->arg_size() >= 1) {
          Value* This = stripToContainerSource(CB->getArgOperand(0));
          if (This != M.Inner.PermContainer) {
            LLVM_DEBUG(dbgs() << "[gate2] vector assignment to a container "
                                 "other than the permutation container "
                                 "(dropped output side effect): "
                              << I << "\n");
            return false;
          }
        }
        continue;
      }
      LLVM_DEBUG(dbgs() << "[gate2] unrecognised side-effecting call: " << I
                        << "\n");
      return false;
    }
  }
  return true;
}

// Replacement

// Replace the matched outer+inner loops with a call to @tsp_impl.
// Signature: i32 @tsp_impl(ptr nodes, ptr cost)
static bool performReplacement(const TspMatch& M) {
  BasicBlock* Preheader = M.OuterL->getLoopPreheader();
  BasicBlock* ExitBB = getNormalExitBlock(M.OuterL);
  if (!ExitBB) {
    LLVM_DEBUG(dbgs() << "  [skip] outer loop has multiple normal exits\n");
    return false;
  }

  Value* NodesArg = M.Inner.PermContainer;
  Value* CostArg = M.Inner.CostMatrix;
  if (!NodesArg || !CostArg) {
    LLVM_DEBUG(dbgs() << "  [skip] cannot determine call arguments\n");
    return false;
  }

  Module* Mod = Preheader->getModule();
  LLVMContext& Ctx = Mod->getContext();
  PointerType* PtrTy = PointerType::get(Ctx, 0);

  FunctionType* FTy =
      FunctionType::get(Type::getInt32Ty(Ctx), {PtrTy, PtrTy}, false);
  auto* ImplFn =
      cast<Function>(Mod->getOrInsertFunction("tsp_impl", FTy).getCallee());
  ImplFn->setDoesNotThrow();

  IRBuilder<> Builder(Preheader->getTerminator());
  CallInst* Result =
      Builder.CreateCall(ImplFn, {NodesArg, CostArg}, "tsp_result");

  // Replace uses of MinAcc's post-loop value (a load outside the loop) with
  // the call result; the load itself becomes dead once the loop is erased.
  SmallVector<LoadInst*, 4> ExternalLoads;
  for (User* U : M.MinAcc->users()) {
    auto* LI = dyn_cast<LoadInst>(U);
    if (LI && !M.OuterL->contains(LI->getParent()))
      ExternalLoads.push_back(LI);
  }
  for (LoadInst* LI : ExternalLoads)
    LI->replaceAllUsesWith(Result);

  SmallVector<BasicBlock*, 32> LoopBlocks(M.OuterL->blocks());

  for (BasicBlock* BB : LoopBlocks) {
    for (BasicBlock* Succ : successors(BB)) {
      if (!M.OuterL->contains(Succ))
        Succ->removePredecessor(BB, /*KeepOneInputPHIs=*/false);
    }
  }

  Preheader->getTerminator()->eraseFromParent();
  UncondBrInst::Create(ExitBB, Preheader);

  for (BasicBlock* BB : LoopBlocks)
    BB->dropAllReferences();
  for (BasicBlock* BB : LoopBlocks)
    BB->eraseFromParent();

  for (LoadInst* LI : ExternalLoads)
    if (LI->use_empty())
      LI->eraseFromParent();

  errs() << "  *** replaced TSP loop with call to @tsp_impl\n\n";
  return true;
}

// Reporting
static void printMatch(const TspMatch& M) {
  errs() << "\n  *** TSP pattern matched ***\n";

  errs() << "  -- Scoring loop --\n";
  errs() << "    header      : " << M.Inner.L->getHeader()->getName() << "\n";
  errs() << "    idx iter    : " << *M.Inner.IdxPhi << "\n";
  errs() << "    prev node   : " << *M.Inner.PrevNodePhi << "\n";
  errs() << "    cost acc    : " << *M.Inner.CostAcc << "\n";
  errs() << "    cost[u][v]  : " << *M.Inner.CostAdd << "\n";

  errs() << "  -- Outer loop --\n";
  errs() << "    header      : " << M.OuterL->getHeader()->getName() << "\n";
  errs() << "    min acc     : " << *M.MinAcc << "\n";
  errs() << "    next_perm   : " << *M.NextPermCall << "\n";
  errs() << "    wrap-close  : " << *M.CloseAdd << "\n";
  if (M.MinCallForm)
    errs() << "    min update  : " << *M.MinCallForm << "\n";
  else
    errs() << "    min update  : " << *M.MinCmpForm << "\n";

  errs() << "  -- Inputs --\n";
  errs() << "    nodes       : " << *M.Inner.PermContainer << "\n";
  errs() << "    cost matrix : " << *M.Inner.CostMatrix << "\n";
  errs() << "\n";
}

// Pass
namespace {
struct TspPass : PassInfoMixin<TspPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager& AM) {
    LoopInfo& LI = AM.getResult<LoopAnalysis>(F);
    LLVM_DEBUG(dbgs() << "[TSP] scanning: " << F.getName() << "\n");

    SmallVector<TspMatch, 2> Matches;
    SmallVector<Loop*, 16> AllLoops;
    for (Loop* TopL : LI)
      collectAllLoops(TopL, AllLoops);

    for (Loop* L : AllLoops) {
      auto Scoring = matchTspScoringLoop(L);
      if (!Scoring)
        continue;
      LLVM_DEBUG(dbgs() << "Found TSP scoring loop\n");

      auto Full = matchTsp(*Scoring);
      if (!Full) {
        LLVM_DEBUG(dbgs() << "  [note] scoring loop in "
                          << L->getHeader()->getName()
                          << " has no enclosing permutation loop\n");
        continue;
      }

      bool Dup = false;
      for (auto& E : Matches)
        if (E.OuterL == Full->OuterL) {
          Dup = true;
          break;
        }
      if (!Dup)
        Matches.push_back(*Full);
    }

    if (Matches.empty()) {
      LLVM_DEBUG(dbgs() << "  no TSP pattern found.\n");
      return PreservedAnalyses::all();
    }

    bool Changed = false;
    for (auto& M : Matches) {
      printMatch(M);
      if (!checkLiveOuts(M)) {
        errs() << "  [skip replacement] unexpected register live-outs\n";
        continue;
      }
      if (!checkSideEffects(M)) {
        errs() << "  [skip replacement] unaccounted side effects\n";
        continue;
      }
      if (performReplacement(M)) {
        Changed = true;
        break;  // LoopInfo is stale after modification
      }
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
}  // namespace

void registerTspPass(PassBuilder& PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager& FPM,
         ArrayRef<PassBuilder::PipelineElement>) -> bool {
        if (Name == "tsp-pass") {
          FPM.addPass(TspPass());
          return true;
        }
        return false;
      });
}
