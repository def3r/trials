#!/usr/bin/env bash
# run_factor_integration.sh - full pipeline: compile → pass → link → run
#
# Mirrors run_clique_integration.sh for factor-pass. Uses the reduced
# canonicalization pipeline (no loop-simplify/lcssa/indvars) confirmed
# correct in analysis/factor/factor.md §1c -- factor-pass matches a plain
# nested loop shape, and indvars' llvm.smax clamp on the outer loop bound
# breaks the NArg check the same way it does for clique-pass.
#
# Steps:
#   1. clang: C++ source → unoptimised LLVM IR
#   2. opt:   canonicalisation passes + factor-pass → transformed IR
#              (bruteForceFactor's loop nest replaced with a call to
#              @factor_impl)
#   3. clang: transformed IR → object file
#   4. link:  object + libc2cudaq.a → binary using cudaq's clang-16 directly
#   5. run the binary
#
# See run_integration.sh for why system clang++ (not nvq++) is used for
# steps 1-3 and 5 (libstdc++ IR shape vs. libc++ cudaq runtime). Full
# cudaq runtime linking is still required even though factor_impl itself
# has no quantum dependency -- bridge.cpp is one translation unit, and
# pulling in factor_impl's object code pulls in the whole object file,
# which references c2q_maxcut/c2q_kcolor/c2q_clique et al. from the other
# bridge functions in the same file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
C2CUDAQ_ROOT="/home/def3r/def3r/SIGSEGV/QuantumComp/c2cudaq"

CLANG="${CLANG:-clang++}"
OPT="${OPT:-/home/def3r/def3r/llvm-project/install/bin/opt}"
CUDAQ_DIR="${CUDAQ_DIR:-/home/def3r/.cudaq}"

# Backend detection -- mirrors nvq++'s own query_gpu() + default-target
# logic exactly (nvq++ script, ~line 156/423-429): prefer the GPU-backed
# nvidia target (cuStateVec FP32) when a GPU is present and the backend
# library exists, otherwise fall back to CPU-backed qpp. Without this,
# a hardcoded -lnvqir-qpp silently runs every kernel call on CPU even on
# a GPU-equipped machine -- confirmed to make a difference: factor's
# n=91 kernel-first call took 46s on GPU vs. 5+ minutes on CPU for the
# identical computation.
gpu_found=false
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L 2>/dev/null | grep -qi "failed\|error"; then
        gpu_found=false
    elif [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 0 ]; then
        gpu_found=true
    fi
fi
NVQIR_BACKEND="qpp"
if [ "$gpu_found" = true ] && [ -f "$CUDAQ_DIR/lib/libnvqir-custatevec-fp32.so" ]; then
    NVQIR_BACKEND="custatevec-fp32"
fi

PLUGIN="$PROJECT_ROOT/build/MinPass.so"
LIB="$C2CUDAQ_ROOT/build/libc2cudaq.a"

OPT_PASSES="sroa,mem2reg,simplifycfg,instcombine<no-verify-fixpoint>,simplifycfg,instcombine"

SRC="$SCRIPT_DIR/factor_e2e.cpp"
BUILD="$SCRIPT_DIR/build"
mkdir -p "$BUILD"

echo "=========================================="
echo " Compiling "
echo "=========================================="
echo ""

# Step 1: C++ → unoptimised IR
echo "[1/5] clang: source → LLVM IR"
"$CLANG" -S -emit-llvm -O0 -fno-inline \
    -Xclang -disable-O0-optnone \
    -fno-discard-value-names \
    -std=c++20 \
    "$SRC" -o "$BUILD/factor_e2e.ll" 2>&1
echo "      $BUILD/factor_e2e.ll"

# Step 2: canonicalization passes + factor-pass
echo ""
echo "[2/5] opt: canonicalization passes → canon IR"
"$OPT" -passes="$OPT_PASSES" \
    "$BUILD/factor_e2e.ll" -S -o "$BUILD/factor_e2e_canon.ll" 2>&1
echo "      $BUILD/factor_e2e_canon.ll"

echo ""
echo "[3/5] opt: factor-pass → transformed IR"
"$OPT" -load-pass-plugin "$PLUGIN" \
    -passes=factor-pass \
    "$BUILD/factor_e2e_canon.ll" -S -o "$BUILD/factor_e2e_transformed.ll" 2>&1
echo "      $BUILD/factor_e2e_transformed.ll"

echo ""
if grep -q "call i1 @factor_impl" "$BUILD/factor_e2e_transformed.ll"; then
    echo "      ✓  @factor_impl injected — pass fired"
    echo ""
    echo "      Call site:"
    grep "factor_impl" "$BUILD/factor_e2e_transformed.ll" \
        | grep -v "^;" | grep -v "@\.str" | sed 's/^/        /'
else
    echo "      ✗  @factor_impl NOT found — pass did not fire"
    exit 1
fi

# Step 4: transformed IR → object
echo ""
echo "[4/5] clang: transformed IR → object"
"$CLANG" -c "$BUILD/factor_e2e_transformed.ll" -o "$BUILD/factor_e2e.o" 2>&1
echo "      $BUILD/factor_e2e.o"

# Step 5: link with cudaq runtime
echo ""
echo "[5/5] clang: link with libc2cudaq.a + cudaq runtime"
"$CLANG" \
    -Wl,-rpath,"$CUDAQ_DIR/lib" \
    -L"$CUDAQ_DIR/lib" \
    "$BUILD/factor_e2e.o" \
    "$LIB" \
    -lc++ -lcudaq -lcudaq-logger -lcudaq-common \
    -lcudaq-ensmallen -lcudaq-nlopt -lcudaq-operator \
    -lcudaq-mlir-runtime -lcudaq-builder \
    -lcudaq-em-default -lcudaq-platform-default \
    -lnvqir -lnvqir-"$NVQIR_BACKEND" \
    -o "$BUILD/factor_e2e" 2>&1
echo "      $BUILD/factor_e2e"

# Run
echo ""
echo "=========================================="
echo " Running"
echo "=========================================="
echo ""
"$BUILD/factor_e2e"
