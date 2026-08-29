#!/usr/bin/env bash
# run_integration.sh - full pipeline: compile → pass → link → run
#
# Steps:
#   1. clang: C++ source → unoptimised LLVM IR
#   2. opt:   optimisation passes + maxcut-pass → transformed IR
#              (brute-force loops replaced with @maxcut_impl)
#   3. clang: transformed IR → object file
#   4. link:  object + libc2cudaq.a → binary using cudaq's clang-16 directly
#   5. run the binary
#
# Why not nvq++?
#   nvq++ is just a shell wrapper around cudaq's clang-16 that adds -L/-l flags
#   for the cudaq runtime (lcudaq, lnvqir, lnvqir-qpp, etc.).  We invoke
#   cudaq's clang-16 directly so we can control the exact link line without
#   the nvq++ wrapper adding unwanted flags.
#
# Why system clang++ to link (not cudaq's clang-16)?
#   The pass was built to match libstdc++ IR (__gnu_cxx::__normal_iterator).
#   libc++ (cudaq's runtime) uses __wrap_iter and the pass doesn't fire on it.
#   So we compile with system clang++ (libstdc++) for steps 1-3, then link
#   with system clang++ which implicitly brings in libstdc++.so — satisfying
#   all libstdc++ symbols while also pulling in the cudaq runtime libs via -L/-l.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
C2CUDAQ_ROOT="/home/def3r/def3r/SIGSEGV/QuantumComp/c2cudaq"

CLANG="${CLANG:-clang++}"
OPT="${OPT:-/home/def3r/def3r/llvm-project/install/bin/opt}"
LLVM_EXTRACT="${LLVM_EXTRACT:-/home/def3r/def3r/llvm-project/install/bin/llvm-extract}"
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

# Optional --func=<name>: after each opt step, dump the CFG of that function
# as a dot file in the build directory so it can be viewed with xdot.
FUNC=""
for _arg in "$@"; do
    case "$_arg" in
        --func=*) FUNC="${_arg#--func=}" ;;
        *) echo "Unknown argument: $_arg"; exit 1 ;;
    esac
done

# dump_dot_cfg <ll_file> <label>
# Extracts the CFG of $FUNC from <ll_file> as <build>/<label>_<func>.dot.
# No-op when --func was not passed.
dump_dot_cfg() {
    local ll_file="$1" label="$2"
    [[ -z "$FUNC" ]] && return 0

    # Find the IR symbol whose demangled base name matches $FUNC.
    local sym=''
    while IFS= read -r s; do
        local base
        base=$(c++filt "$s" 2>/dev/null | sed 's/(.*//')
        if [[ "$base" == "$FUNC" ]]; then sym="$s"; break; fi
    done < <(grep -E '^define ' "$ll_file" | grep -oE '@[_a-zA-Z0-9.]+' | tr -d '@')

    if [[ -z "$sym" ]]; then
        echo "      [dot-cfg] no function '$FUNC' in $(basename "$ll_file")"
        return 0
    fi

    local ex_ll="$BUILD/${label}_${FUNC}_ex.ll"
    local dot_out="$BUILD/${label}_${FUNC}.dot"

    # Isolate the function so only one dot file is generated.
    "$LLVM_EXTRACT" -func="$sym" "$ll_file" -S -o "$ex_ll" 2>/dev/null

    # dot-cfg writes <prefix>.<sym>.dot in CWD; use label as prefix so each
    # step gets a distinct filename.
    (cd "$BUILD" && "$OPT" -passes=dot-cfg \
        -cfg-dot-filename-prefix="$label" \
        -disable-output "$(basename "$ex_ll")" 2>/dev/null)

    if [[ -f "$BUILD/${label}.${sym}.dot" ]]; then
        mv "$BUILD/${label}.${sym}.dot" "$dot_out"
        echo "      [dot-cfg] $dot_out"
        echo "      [dot-cfg] view: xdot $dot_out"
    else
        echo "      [dot-cfg] warning: expected ${label}.${sym}.dot was not generated"
    fi
}

PLUGIN="$PROJECT_ROOT/build/MinPass.so"
LIB="$C2CUDAQ_ROOT/build/libc2cudaq.a"

OPT_PASSES="sroa,mem2reg,loop-simplify,lcssa,indvars,simplifycfg,instcombine<no-verify-fixpoint>,simplifycfg,instcombine<no-verify-fixpoint>"

SRC="$SCRIPT_DIR/maxcut_e2e.cpp"
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
    "$SRC" -o "$BUILD/maxcut_e2e.ll" 2>&1
echo "      $BUILD/maxcut_e2e.ll"
dump_dot_cfg "$BUILD/maxcut_e2e.ll" "step1_unopt"

# Step 2: canonicalization passes only (no maxcut-pass)
echo ""
echo "[2/5] opt: canonicalization passes → canon IR"
"$OPT" -passes="$OPT_PASSES" \
    "$BUILD/maxcut_e2e.ll" -S -o "$BUILD/maxcut_e2e_canon.ll" 2>&1
echo "      $BUILD/maxcut_e2e_canon.ll"
dump_dot_cfg "$BUILD/maxcut_e2e_canon.ll" "step2_canon"

# Step 3: maxcut-pass on already-canonicalised IR
echo ""
echo "[3/5] opt: maxcut-pass → transformed IR"
"$OPT" -load-pass-plugin "$PLUGIN" \
    -passes=maxcut-pass \
    "$BUILD/maxcut_e2e_canon.ll" -S -o "$BUILD/maxcut_e2e_transformed.ll" 2>&1
echo "      $BUILD/maxcut_e2e_transformed.ll"
dump_dot_cfg "$BUILD/maxcut_e2e_transformed.ll" "step3_transformed"

echo ""
if grep -q "call i32 @maxcut_impl" "$BUILD/maxcut_e2e_transformed.ll"; then
    echo "      ✓  @maxcut_impl injected — pass fired"
    echo ""
    echo "      Call site:"
    grep "maxcut_impl" "$BUILD/maxcut_e2e_transformed.ll" \
        | grep -v "^;" | grep -v "@\.str" | sed 's/^/        /'
else
    echo "      ✗  @maxcut_impl NOT found — pass did not fire"
    exit 1
fi

# Step 4: transformed IR → object
echo ""
echo "[4/5] clang: transformed IR → object"
"$CLANG" -c "$BUILD/maxcut_e2e_transformed.ll" -o "$BUILD/maxcut_e2e.o" 2>&1
echo "      $BUILD/maxcut_e2e.o"

# Step 5: link with cudaq runtime
echo ""
echo "[5/5] clang: link with libc2cudaq.a + cudaq runtime"
"$CLANG" \
    -Wl,-rpath,"$CUDAQ_DIR/lib" \
    -L"$CUDAQ_DIR/lib" \
    "$BUILD/maxcut_e2e.o" \
    "$LIB" \
    -lc++ -lcudaq -lcudaq-logger -lcudaq-common \
    -lcudaq-ensmallen -lcudaq-nlopt -lcudaq-operator \
    -lcudaq-mlir-runtime -lcudaq-builder \
    -lcudaq-em-default -lcudaq-platform-default \
    -lnvqir -lnvqir-"$NVQIR_BACKEND" \
    -o "$BUILD/maxcut_e2e" 2>&1
echo "      $BUILD/maxcut_e2e"

# Run
echo ""
echo "=========================================="
echo " Running"
echo "=========================================="
echo ""
"$BUILD/maxcut_e2e"
