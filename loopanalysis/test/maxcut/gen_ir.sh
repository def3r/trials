#!/usr/bin/env bash
# gen_ir.sh - Regenerate all _opt.ll files via update.py, then run
# maxcut-cpp-pass on each one and report whether it fires.
#
# Expected outcome is read from the CHECK lines already embedded in each
# _opt.ll by update.py, so no separate expect-list is needed here:
#   ; CHECK: ...maxcut_impl...  → DETECT
#   ; CHECK-NOT: maxcut_impl    → REJECT
#   ; XFAIL: ...               → XFAIL (known limitation, not detected)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT="${OPT:-/home/def3r/def3r/llvm-project/install/bin/opt}"
LLVM_EXTRACT="${LLVM_EXTRACT:-/home/def3r/def3r/llvm-project/install/bin/llvm-extract}"
PLUGIN="$SCRIPT_DIR/../../build/MinPass.so"

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
RST='\033[0m'

cd "$SCRIPT_DIR"

# Step 1: (re)generate all _opt.ll files
echo "Regenerating _opt.ll files..."
LLVM_EXTRACT="$LLVM_EXTRACT" python3 update.py
echo ""

# Step 2: run the pass on each _opt.ll and report
echo "=============================================="
echo " MaxCut Pass Results"
echo "=============================================="
echo ""

pass_count=0
fail_count=0
xfail_count=0

for opt_ll in $(ls *_opt.ll | sort); do
    base="${opt_ll%_opt.ll}"

    # Derive expected outcome from the embedded CHECK directives.
    if grep -q '^; XFAIL' "$opt_ll"; then
        expect="XFAIL"
    elif grep -q '^; CHECK:' "$opt_ll"; then
        expect="DETECT"
    else
        expect="REJECT"
    fi

    # Run the pass; capture stderr (all pass output goes to errs()).
    output=$("$OPT" -load-pass-plugin "$PLUGIN" \
                    -passes=maxcut-pass \
                    -disable-output \
                    "$opt_ll" 2>&1) || true
    fired=0
    echo "$output" | grep -q "replaced MaxCut loops" && fired=1

    printf "%-28s  " "${base}.cpp"
    if [[ "$expect" == "DETECT" ]]; then
        if [[ $fired -eq 1 ]]; then
            printf "${GRN}PASS${RST}  detected\n"
            pass_count=$(( pass_count + 1 ))
        else
            printf "${RED}FAIL${RST}  not detected (should fire)\n"
            fail_count=$(( fail_count + 1 ))
        fi
    elif [[ "$expect" == "REJECT" ]]; then
        if [[ $fired -eq 0 ]]; then
            printf "${GRN}PASS${RST}  not detected\n"
            pass_count=$(( pass_count + 1 ))
        else
            printf "${RED}FAIL${RST}  incorrectly detected (should not fire)\n"
            fail_count=$(( fail_count + 1 ))
        fi
    elif [[ "$expect" == "XFAIL" ]]; then
        if [[ $fired -eq 0 ]]; then
            printf "${YLW}XFAIL${RST} not detected (known limitation)\n"
            xfail_count=$(( xfail_count + 1 ))
        else
            printf "${GRN}FIXED${RST} now detected — update CHECK lines and remove XFAIL\n"
            pass_count=$(( pass_count + 1 ))
        fi
    fi
done

echo ""
echo "=============================================="
echo " ${pass_count} passed  ${fail_count} failed  ${xfail_count} xfail"
echo "=============================================="
[[ $fail_count -eq 0 ]]
