#!/usr/bin/env python3
"""
update.py — Recompile Factor test cases and embed CHECK directives into _opt.ll.

Usage:
  python3 update.py             # recompile all registered test cases
  python3 update.py basic.cpp   # recompile one file

Same mechanism as test/clique/update.py: factor-pass is a plain
FunctionPass, but (like clique-pass) each test's .cpp is small enough to
compile and canonicalize directly with no llvm-extract step -- there's
only ever one function of interest per test file here, so no
FUNC_GREP-style function-name registration is needed either.

CHECK lines are written as // CHECK: / // CHECK-NOT: / // XFAIL: comments
in the .cpp source file. This script extracts them and prepends them (as
; CHECK: / ; XFAIL: lines) to the generated _opt.ll, together with the
; RUN: directive that lit uses to drive the test.

After running:
  python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/factor/
"""

import os
import re
import subprocess
import sys

TESTS_DIR  = os.path.dirname(os.path.abspath(__file__))
CLANG      = os.environ.get('CLANG', 'clang')
OPT        = os.environ.get('OPT',   '/home/def3r/def3r/llvm-project/install/bin/opt')
OPT_PASSES = ('sroa,mem2reg,simplifycfg,instcombine<no-verify-fixpoint>,'
             'simplifycfg,instcombine')

RUN_LINE = ('; RUN: %opt -load-pass-plugin %plugin '
            '-passes=factor-pass -S %s 2>&1 | %FileCheck %s\n')

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


def compile_pipeline(cpp_path: str, base: str) -> str:
    """clang → opt; returns path to generated _opt.ll. No llvm-extract."""
    ll     = base + '.ll'
    opt_ll = base + '_opt.ll'

    run([CLANG, '-S', '-emit-llvm', '-O0', '-fno-inline',
         '-Xclang', '-disable-O0-optnone', '-fno-discard-value-names',
         cpp_path, '-o', ll])
    run([OPT, f'-passes={OPT_PASSES}', ll, '-S', '-o', opt_ll])
    return opt_ll


def classify(checks: list[str]) -> str:
    if any('XFAIL' in c for c in checks):
        return 'XFAIL'
    if any('CHECK-NOT' not in c and 'CHECK:' in c for c in checks):
        return 'DETECT'
    return 'REJECT'


def process(cpp_file: str) -> bool:
    name = cpp_file.removesuffix('.cpp')
    cpp_path = os.path.join(TESTS_DIR, cpp_file)
    base     = os.path.join(TESTS_DIR, name)

    try:
        opt_ll = compile_pipeline(cpp_path, base)
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
