#!/usr/bin/env python3
"""
update.py — Recompile KColor test cases and embed CHECK directives into _opt.ll.

Usage:
  python3 update.py             # recompile all registered test cases
  python3 update.py basic.cpp   # recompile one file

Unlike maxcut/tsp's update.py (which extract a single named function),
kcolor-pass matches are inter-procedural: the recursive solve()-shaped
function and the top-level call site that invokes it may be (and in the
reference shape, are) different functions. Each test's functions share a
name prefix (registered in FUNC_GREP below) so a single llvm-extract
--rfunc=<regex> pulls out every function the test needs as a real
definition, leaving unrelated declarations behind.

CHECK lines are written as // CHECK: / // CHECK-NOT: / // XFAIL: comments
in the .cpp source file. This script extracts them and prepends them (as
; CHECK: / ; XFAIL: lines) to the generated _opt.ll, together with the
; RUN: directive that lit uses to drive the test.

After running:
  python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/kcolor/
"""

import os
import re
import subprocess
import sys

TESTS_DIR    = os.path.dirname(os.path.abspath(__file__))
CLANG        = os.environ.get('CLANG',        'clang')
OPT          = os.environ.get('OPT',          '/home/def3r/def3r/llvm-project/install/bin/opt')
LLVM_EXTRACT = os.environ.get('LLVM_EXTRACT', 'llvm-extract')
OPT_PASSES   = 'sroa,mem2reg,simplifycfg,instcombine,simplifycfg,instcombine'

RUN_LINE = ('; RUN: %opt -load-pass-plugin %plugin '
            '-passes=kcolor-pass -S %s 2>&1 | %FileCheck %s\n')

# Maps .cpp basename (without extension) → regex passed to llvm-extract
# --rfunc=, matching every function (mangled name) the test needs kept as a
# real definition (guard fn, solve fn, entry fn(s), ...).
FUNC_GREP = {
    'basic':                     r'kc_basic_',
    'inner_log':                 r'kc_innerlog_',
    'guard_log':                 r'kc_guardlog_',
    'extern_memset':             r'kc_externmemset_',
    'assign_backtrack_mismatch': r'kc_mismatch_',
    'uint_graph':                r'kc_uint_',
    'wide_bounds':                r'kc_wide_',
    'multiple_self_calls':       r'kc_multiself_',
    'no_recursion':              r'kc_norecur_',
    'non_unit_recursion':        r'kc_nonunit_',
    'swapped_passthrough':       r'kc_swapped_',
    'wrong_base_case':           r'kc_wrongbase_',
    'inlined_guard':             r'kc_inlined_',
    'guard_calls_solve':         r'kc_crossrecur_',
    'no_backtrack':              r'kc_nobacktrack_',
    'no_top_level_call':         r'kc_notoplevel_',
    'multi_call_site':           r'kc_multicall_',
    'plain_call_site':           r'kc_plaincall_',
    'wrong_arg_count':           r'kc_argcount_',
    'int_return':                r'kc_intreturn_',
    'nqueens_lookalike':         r'kc_nqueens_',
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


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def compile_pipeline(cpp_path: str, base: str, pattern: str) -> str:
    """clang → llvm-extract (regex, multi-func) → opt; returns _opt.ll path."""
    ll     = base + '.ll'
    ex_ll  = base + '_ex.ll'
    opt_ll = base + '_opt.ll'

    run([CLANG, '-S', '-emit-llvm', '-O0', '-fno-inline',
         '-Xclang', '-disable-O0-optnone', '-fno-discard-value-names',
         cpp_path, '-o', ll])

    run([LLVM_EXTRACT, f'-rfunc={pattern}', ll, '-S', '-o', ex_ll])
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
