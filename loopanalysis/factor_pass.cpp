#include <optional>
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
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

#define DEBUG_TYPE "factor-cpp"  // -debug-only=factor-cpp

using namespace llvm;

// Data structures
//
// Unlike the other four passes, this one targets a genuinely unstructured
// brute-force search, not a smarter classical algorithm -- because that's
// what c2q_factor's Grover oracle actually accelerates (see
// analysis/factor/factor.md): a QFT-multiplier phase-flips states where
// a*b == n over superposed pairs (a, b), with no sqrt(n) shortcut built
// in. The classical equivalent is a plain nested double loop testing every
// (a, b) pair in [start, n) x [start, n), not a single-loop trial-division
// scan. That shape is fully SSA after mem2reg -- both loop counters stay
// phis (no std::min/std::max-by-reference forcing an alloca the way every
// other pass's accumulator does), so this matcher works entirely on phi
// backedges and icmp operands, with no memory-resident accumulator to
// trace at all.
struct FactorMatch {
  Function* F;
  Loop* OuterL;   // walks candidate 'a'
  Loop* InnerL;   // walks candidate 'b', nested directly inside OuterL
  PHINode* APhi;
  PHINode* BPhi;
  Argument* NArg;  // shared bound for both loops AND the product compare
  BinaryOperator* Mul;    // a * b (or b * a)
  ICmpInst* ProdCmp;      // icmp eq Mul, NArg
  BasicBlock* ExitBB;     // shared merge block: found-edge and exhausted-edge
  PHINode* AResult;       // [APhi, found] / [1, exhausted]
  PHINode* BResult;       // [BPhi, found] / [NArg, exhausted]
  StoreInst* StoreA;
  StoreInst* StoreB;
  Argument* OutAArg;
  Argument* OutBArg;
  ReturnInst* Ret;
};

// Helpers (duplicated per the established per-pass convention).

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

static Value* getCondBrCondition(BasicBlock* BB) {
  if (auto* CBR = dyn_cast<CondBrInst>(BB->getTerminator()))
    return CBR->getCondition();
  return nullptr;
}

// Returns true iff V is structurally `add nsw/nuw Base, 1`.
static bool isAddOne(Value* V, Value* Base) {
  auto* BO = dyn_cast<BinaryOperator>(V);
  if (!BO || BO->getOpcode() != Instruction::Add)
    return false;
  Value *Op0 = BO->getOperand(0), *Op1 = BO->getOperand(1);
  auto* C = dyn_cast<ConstantInt>(Op1);
  return Op0 == Base && C && C->isOne();
}

static bool isLtPredicate(ICmpInst::Predicate P) {
  return P == ICmpInst::ICMP_SLT || P == ICmpInst::ICMP_ULT;
}

// Match: a two-level nested loop searching pairs (a, b) in [c0, n) x [c1,
// n) for a*b == n, storing the match (or the fallback pair (1, n)) into
// two output parameters and returning whether a match was found.
static std::optional<FactorMatch> matchFactor(Function& F, LoopInfo& LI) {
  if (F.isDeclaration() || F.arg_size() != 3)
    return std::nullopt;
  if (!F.getReturnType()->isIntegerTy(1))
    return std::nullopt;

  for (Loop* OuterL : LI) {
    if (OuterL->getSubLoops().size() != 1)
      continue;
    Loop* InnerL = OuterL->getSubLoops()[0];
    if (!InnerL->getSubLoops().empty())
      continue;

    BasicBlock* OuterHeader = OuterL->getHeader();
    BasicBlock* OuterPreheader = OuterL->getLoopPreheader();
    BasicBlock* OuterLatch = OuterL->getLoopLatch();
    BasicBlock* InnerHeader = InnerL->getHeader();
    // Not getLoopPreheader(): that requires the single external predecessor
    // to have exactly one successor, but here it's the OUTER header itself
    // -- which necessarily has two (into the inner loop, and out to the
    // exit block). getLoopPredecessor() only requires a unique external
    // predecessor, which is exactly the relationship this pattern has.
    BasicBlock* InnerPreheader = InnerL->getLoopPredecessor();
    BasicBlock* InnerLatch = InnerL->getLoopLatch();
    if (!OuterPreheader || !OuterLatch || !InnerPreheader || !InnerLatch)
      continue;

    // 1: the inner loop must be entered directly from the outer header --
    // no init code between "a is in range" and "start scanning b" (mirrors
    // TSP's Phase 2 "scoring loop's preheader is the permutation loop's
    // header" coupling check).
    if (InnerPreheader != OuterHeader)
      continue;

    // 2: outer induction phi -- canonical add-by-1 backedge, constant init.
    PHINode* APhi = nullptr;
    for (PHINode& PN : OuterHeader->phis()) {
      if (isAddOne(PN.getIncomingValueForBlock(OuterLatch), &PN)) {
        if (APhi) {
          APhi = nullptr;
          break;
        }
        APhi = &PN;
      }
    }
    if (!APhi ||
        !isa<ConstantInt>(APhi->getIncomingValueForBlock(OuterPreheader)))
      continue;

    // 3: inner induction phi -- same shape, reset from the outer header
    // every outer iteration.
    PHINode* BPhi = nullptr;
    for (PHINode& PN : InnerHeader->phis()) {
      if (isAddOne(PN.getIncomingValueForBlock(InnerLatch), &PN)) {
        if (BPhi) {
          BPhi = nullptr;
          break;
        }
        BPhi = &PN;
      }
    }
    if (!BPhi || !isa<ConstantInt>(BPhi->getIncomingValueForBlock(OuterHeader)))
      continue;

    // 4: outer bound -- icmp slt/ult APhi, <Argument N>. performReplacement
    // emits @factor_impl with a fixed i32 N parameter, so reject rather
    // than build a mismatched call (same lesson as every other pass's NArg
    // type check).
    auto* OuterCmp =
        dyn_cast_or_null<ICmpInst>(getCondBrCondition(OuterHeader));
    if (!OuterCmp || !isLtPredicate(OuterCmp->getPredicate()))
      continue;
    Argument* NArg = nullptr;
    if (OuterCmp->getOperand(0) == APhi)
      NArg = dyn_cast<Argument>(OuterCmp->getOperand(1));
    else if (OuterCmp->getOperand(1) == APhi)
      NArg = dyn_cast<Argument>(OuterCmp->getOperand(0));
    if (!NArg || NArg->getParent() != &F || !NArg->getType()->isIntegerTy(32))
      continue;
    auto* OuterBI = dyn_cast<CondBrInst>(OuterHeader->getTerminator());
    if (!OuterBI || OuterBI->getSuccessor(0) != InnerHeader)
      continue;
    BasicBlock* ExitBB = OuterBI->getSuccessor(1);
    if (OuterL->contains(ExitBB))
      continue;

    // 5: inner bound -- icmp slt/ult BPhi, the SAME NArg (not just some
    // argument -- the identical one). This double identity is what says
    // "this searches the full n x n space" rather than some unrelated
    // double-loop shape.
    auto* InnerCmp =
        dyn_cast_or_null<ICmpInst>(getCondBrCondition(InnerHeader));
    if (!InnerCmp || !isLtPredicate(InnerCmp->getPredicate()))
      continue;
    bool InnerBoundOk =
        (InnerCmp->getOperand(0) == BPhi && InnerCmp->getOperand(1) == NArg) ||
        (InnerCmp->getOperand(1) == BPhi && InnerCmp->getOperand(0) == NArg);
    if (!InnerBoundOk)
      continue;
    auto* InnerBI = dyn_cast<CondBrInst>(InnerHeader->getTerminator());
    if (!InnerBI)
      continue;
    BasicBlock* InnerBodyBB = InnerBI->getSuccessor(0);
    BasicBlock* InnerExitBB = InnerBI->getSuccessor(1);
    if (!InnerL->contains(InnerBodyBB) || InnerExitBB != OuterLatch)
      continue;

    // 6: body -- mul(APhi, BPhi) in either operand order, compared eq
    // against NArg a third time; branch on match exits straight to ExitBB.
    BinaryOperator* Mul = nullptr;
    for (Instruction& I : *InnerBodyBB) {
      auto* BO = dyn_cast<BinaryOperator>(&I);
      if (!BO || BO->getOpcode() != Instruction::Mul)
        continue;
      bool Ok = (BO->getOperand(0) == APhi && BO->getOperand(1) == BPhi) ||
                (BO->getOperand(1) == APhi && BO->getOperand(0) == BPhi);
      if (!Ok)
        continue;
      Mul = BO;
      break;
    }
    if (!Mul)
      continue;
    ICmpInst* ProdCmp = nullptr;
    for (User* U : Mul->users()) {
      auto* Cmp = dyn_cast<ICmpInst>(U);
      if (!Cmp || Cmp->getPredicate() != ICmpInst::ICMP_EQ)
        continue;
      Value* Other = Cmp->getOperand(0) == Mul ? Cmp->getOperand(1)
                                               : Cmp->getOperand(0);
      if (Other != NArg)
        continue;
      ProdCmp = Cmp;
      break;
    }
    if (!ProdCmp || ProdCmp->getParent() != InnerBodyBB)
      continue;
    auto* BodyBI = dyn_cast<CondBrInst>(InnerBodyBB->getTerminator());
    if (!BodyBI || BodyBI->getCondition() != ProdCmp)
      continue;
    if (BodyBI->getSuccessor(0) != ExitBB ||
        !InnerL->contains(BodyBI->getSuccessor(1)))
      continue;

    // 7: merge phis at ExitBB -- kept tight per design decision: the
    // "not found" edge must carry exactly (1, NArg), not any constant
    // pair, matching decode_factors' expected prime-fallback convention.
    PHINode *AResult = nullptr, *BResult = nullptr;
    for (PHINode& PN : ExitBB->phis()) {
      Value* FoundVal = PN.getIncomingValueForBlock(InnerBodyBB);
      Value* FallbackVal = PN.getIncomingValueForBlock(OuterHeader);
      if (FoundVal == APhi) {
        auto* C = dyn_cast<ConstantInt>(FallbackVal);
        if (C && C->isOne())
          AResult = &PN;
      } else if (FoundVal == BPhi) {
        if (FallbackVal == NArg)
          BResult = &PN;
      }
    }
    if (!AResult || !BResult)
      continue;

    // 8: exactly two stores, into two distinct pointer-typed parameters.
    StoreInst *StoreA = nullptr, *StoreB = nullptr;
    Argument *OutAArg = nullptr, *OutBArg = nullptr;
    for (Instruction& I : *ExitBB) {
      auto* SI = dyn_cast<StoreInst>(&I);
      if (!SI)
        continue;
      auto* Ptr = dyn_cast_or_null<Argument>(
          stripToContainerSource(SI->getPointerOperand()));
      if (!Ptr || Ptr->getParent() != &F || Ptr == NArg)
        continue;
      if (SI->getValueOperand() == AResult) {
        StoreA = SI;
        OutAArg = Ptr;
      } else if (SI->getValueOperand() == BResult) {
        StoreB = SI;
        OutBArg = Ptr;
      }
    }
    if (!StoreA || !StoreB || OutAArg == OutBArg)
      continue;

    // 9: the merge phis must have no users besides their own store --
    // performReplacement erases them outright.
    if (!AResult->hasOneUser() || !BResult->hasOneUser())
      continue;

    // 10: ExitBB must contain nothing beyond the recognised shape: the two
    // phis, the two stores, and a terminating ret. Anything else (e.g. a
    // logging call) is an unaccounted side effect.
    auto* Ret = dyn_cast<ReturnInst>(ExitBB->getTerminator());
    if (!Ret)
      continue;
    bool ExitBBClean = true;
    for (Instruction& I : *ExitBB) {
      if (&I == AResult || &I == BResult || &I == StoreA || &I == StoreB ||
          &I == Ret)
        continue;
      ExitBBClean = false;
      break;
    }
    if (!ExitBBClean)
      continue;

    return FactorMatch{&F,       OuterL, InnerL,  APhi,    BPhi,   NArg,
                       Mul,      ProdCmp, ExitBB, AResult, BResult,
                       StoreA,   StoreB,  OutAArg, OutBArg, Ret};
  }
  return std::nullopt;
}

// Gate: no calls or stores anywhere inside the matched loop nest itself --
// this pattern is pure integer arithmetic in registers; any side effect
// beyond the two recognised output stores in ExitBB (outside the loop
// proper) is unaccounted for.
static bool checkSideEffects(const FactorMatch& M) {
  for (BasicBlock* BB : M.OuterL->blocks()) {
    for (Instruction& I : *BB) {
      if (isa<CallBase>(&I)) {
        LLVM_DEBUG(dbgs() << "[gate] unexpected call inside matched loop: "
                          << I << "\n");
        return false;
      }
      if (isa<StoreInst>(&I)) {
        LLVM_DEBUG(dbgs() << "[gate] unexpected store inside matched loop: "
                          << I << "\n");
        return false;
      }
    }
  }
  return true;
}

// Replace the matched loop nest with a call to @factor_impl.
// Signature: i1 @factor_impl(i32 n, ptr outA, ptr outB)
static bool performReplacement(const FactorMatch& M) {
  BasicBlock* Preheader = M.OuterL->getLoopPreheader();
  if (!Preheader)
    return false;

  Module* Mod = Preheader->getModule();
  LLVMContext& Ctx = Mod->getContext();
  PointerType* PtrTy = PointerType::get(Ctx, 0);
  Type* I32Ty = Type::getInt32Ty(Ctx);
  Type* I1Ty = Type::getInt1Ty(Ctx);

  FunctionType* FTy = FunctionType::get(I1Ty, {I32Ty, PtrTy, PtrTy}, false);
  auto* ImplFn =
      cast<Function>(Mod->getOrInsertFunction("factor_impl", FTy).getCallee());
  ImplFn->setDoesNotThrow();

  IRBuilder<> Builder(Preheader->getTerminator());
  CallInst* Result = Builder.CreateCall(
      ImplFn, {M.NArg, M.OutAArg, M.OutBArg}, "factor_result");

  // ExitBB's stores/phis compute the function's own two outputs, and its
  // `ret` reuses a loop-derived condition value -- factor_impl now owns
  // all three, so strip the originals rather than RAUW a single value the
  // way tsp_pass.cpp's single MinAcc load does.
  IRBuilder<> RetBuilder(M.Ret);
  RetBuilder.CreateRet(Result);
  M.Ret->eraseFromParent();
  M.StoreA->eraseFromParent();
  M.StoreB->eraseFromParent();
  M.AResult->eraseFromParent();
  M.BResult->eraseFromParent();

  SmallVector<BasicBlock*, 8> LoopBlocks(M.OuterL->blocks());
  for (BasicBlock* BB : LoopBlocks) {
    for (BasicBlock* Succ : successors(BB)) {
      if (!M.OuterL->contains(Succ))
        Succ->removePredecessor(BB, /*KeepOneInputPHIs=*/false);
    }
  }

  Preheader->getTerminator()->eraseFromParent();
  UncondBrInst::Create(M.ExitBB, Preheader);

  for (BasicBlock* BB : LoopBlocks)
    BB->dropAllReferences();
  for (BasicBlock* BB : LoopBlocks)
    BB->eraseFromParent();

  errs() << "  *** replaced factor search loop with call to @factor_impl\n\n";
  return true;
}

// Reporting
static void printMatch(const FactorMatch& M) {
  errs() << "\n  *** Brute-force factor-pair search matched ***\n";
  errs() << "    function    : " << M.F->getName() << "\n";
  errs() << "    outer phi   : " << *M.APhi << "\n";
  errs() << "    inner phi   : " << *M.BPhi << "\n";
  errs() << "    product cmp : " << *M.ProdCmp << "\n";
  errs() << "    N           : #" << M.NArg->getArgNo() << "\n";
  errs() << "    outA        : #" << M.OutAArg->getArgNo() << "\n";
  errs() << "    outB        : #" << M.OutBArg->getArgNo() << "\n\n";
}

// Pass
namespace {
struct FactorPass : PassInfoMixin<FactorPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager& AM) {
    LoopInfo& LI = AM.getResult<LoopAnalysis>(F);
    LLVM_DEBUG(dbgs() << "[Factor] scanning: " << F.getName() << "\n");

    auto Match = matchFactor(F, LI);
    if (!Match) {
      LLVM_DEBUG(dbgs() << "  no factor-search pattern found.\n");
      return PreservedAnalyses::all();
    }
    printMatch(*Match);
    if (!checkSideEffects(*Match)) {
      errs() << "  [skip replacement] unaccounted side effects\n";
      return PreservedAnalyses::all();
    }
    if (!performReplacement(*Match))
      return PreservedAnalyses::all();

    return PreservedAnalyses::none();
  }
};
}  // namespace

void registerFactorPass(PassBuilder& PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef Name, FunctionPassManager& FPM,
         ArrayRef<PassBuilder::PipelineElement>) -> bool {
        if (Name == "factor-pass") {
          FPM.addPass(FactorPass());
          return true;
        }
        return false;
      });
}
