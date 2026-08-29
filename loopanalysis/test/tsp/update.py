#!/usr/bin/env python3
"""
update.py — Recompile TSP test cases and embed CHECK directives into _opt.ll.

Usage:
  python3 update.py             # recompile all registered test cases
  python3 update.py basic.cpp   # recompile one file

CHECK lines are written as // CHECK: / // CHECK-NOT: / // XFAIL: comments
in the .cpp source file. This script extracts them and prepends them (as
; CHECK: / ; XFAIL: lines) to the generated _opt.ll, together with the
; RUN: directive that lit uses to drive the test.

After running:
  python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/tsp/
"""

import os
import re
import subprocess
import sys

TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
CLANG        = os.environ.get('CLANG',        'clang')
OPT          = os.environ.get('OPT',          '/home/def3r/def3r/llvm-project/install/bin/opt')
LLVM_EXTRACT = os.environ.get('LLVM_EXTRACT', 'llvm-extract')
OPT_PASSES   = ('sroa,mem2reg,loop-simplify,lcssa,indvars,'
                'simplifycfg,instcombine,simplifycfg,instcombine')

RUN_LINE = ('; RUN: %opt -load-pass-plugin %plugin '
            '-passes=tsp-pass -S %s 2>&1 | %FileCheck %s\n')

# Maps .cpp basename (without extension) → grep pattern for mangled name.
# Pattern is a Python regex matched against mangled symbol names in the .ll.
FUNC_GREP = {
    'basic':                   r'tsp_basic',
    'cost_matrix_mismatch':    r'tsp_cost_mismatch',
    'perm_container_mismatch': r'tsp_perm_mismatch',
    'extern_memset':           r'tsp_extern_memset',
    'inner_log':               r'tsp_inner_log',
    'score_no_min':            r'score_tour_only',
    'sle_compare':             r'tsp_sle',
    'sum_cost':                r'tsp_sum_cost',
    'max_cost':                r'tsp_max_cost',
    'uint_nodes':              r'tsp_uint',
    'wide_cost':               r'tsp_wide_cost',
    'scaled_cost':             r'tsp_scaled',
    'min_cmp_form':            r'tsp_cmp_form',
    'open_path':               r'tsp_open_path',
    'wrong_close_index':       r'tsp_wrong_close',
    'manual_permutation':      r'tsp_manual_perm',
    'best_path_output':        r'tsp_best_path',
}

_CHECK_RE = re.compile(r'\s*//\s*((?:CHECK[^:]*|XFAIL):)(.*)')


def extract_check_lines(cpp_path: str) -> list[str]:
    lines = []
    with open(cpp_path) as f:
        for line in f:
            m = _CHECK_RE.match(line)
            if m:
                lines.append('; ' + m.group(1) + m.group(2) + '\n')
    return lines


def find_mangled(ll_path: str, pattern: str) -> str:
    with open(ll_path) as f:
        content = f.read()
    for name in re.findall(r'@(_Z[^()\s,*\[\]]+)', content):
        if re.search(pattern, name):
            return name
    return ''


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def compile_pipeline(cpp_path: str, base: str, pattern: str) -> str:
    """clang → llvm-extract → opt; returns path to generated _opt.ll."""
    ll     = base + '.ll'
    ex_ll  = base + '_ex.ll'
    opt_ll = base + '_opt.ll'

    run([CLANG, '-S', '-emit-llvm', '-O0', '-fno-inline',
         '-Xclang', '-disable-O0-optnone', '-fno-discard-value-names',
         cpp_path, '-o', ll])

    mangled = find_mangled(ll, pattern)
    if not mangled:
        raise RuntimeError(f'no symbol matching {pattern!r} in {ll}')

    run([LLVM_EXTRACT, f'-func={mangled}', ll, '-S', '-o', ex_ll])
    run([OPT, f'-passes={OPT_PASSES}', ex_ll, '-S', '-o', opt_ll])
    return opt_ll


def classify(checks: list[str]) -> str:
    if any('XFAIL' in c for c in checks):
        return 'XFAIL'
    if any('CHECK-NOT' not in c and 'CHECK:' in c for c in checks):
        return 'DETECT'
    return 'REJECT'


def process(cpp_file: str) -> bool:
    name = cpp_file.removesuffix('.cpp')
    if name not in FUNC_GREP:
        print(f'  SKIP  {cpp_file}  (not in FUNC_GREP — add entry to register)')
        return False

    cpp_path = os.path.join(TESTS_DIR, cpp_file)
    base     = os.path.join(TESTS_DIR, name)

    try:
        opt_ll = compile_pipeline(cpp_path, base, FUNC_GREP[name])
    except subprocess.CalledProcessError as e:
        print(f'  ERROR {cpp_file}: compilation failed ({e})')
        return False
    except RuntimeError as e:
        print(f'  ERROR {cpp_file}: {e}')
        return False

    checks = extract_check_lines(cpp_path)
    if not checks:
        print(f'  WARN  {cpp_file}  (no // CHECK: lines found — '
              f'add them to the .cpp so lit can verify the output)')

    with open(opt_ll) as f:
        ir_body = f.read()

    with open(opt_ll, 'w') as f:
        f.write(RUN_LINE)
        f.write('\n')
        for c in checks:
            f.write(c)
        if checks:
            f.write('\n')
        f.write(ir_body)

    print(f'  OK    {cpp_file}  [{classify(checks)}]')
    return True


def main() -> None:
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = sorted(f for f in os.listdir(TESTS_DIR) if f.endswith('.cpp'))

    print(f'update.py — processing {len(targets)} file(s)\n')
    ok = sum(process(t) for t in targets)
    print(f'\n{ok}/{len(targets)} succeeded.')


if __name__ == '__main__':
    main()
