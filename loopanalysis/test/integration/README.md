# Integration Tests

End-to-end tests, one per pass: compile the reference source → run the
canonicalization pipeline + that pass → confirm the replacement call was
actually emitted → compile the transformed IR → link against
`c2cudaq/build/libc2cudaq.a` + the cudaq runtime → run the binary → check
the actual numeric answer against a known-correct classical result.

This is a different, stronger check than the lit suites in `../maxcut/`,
`../tsp/`, `../kcolor/`, `../clique/`, `../factor/`: those confirm the
*IR transformation* is correct (the right call gets emitted, or correctly
doesn't); these confirm the *linked, running binary* produces the right
*answer* — which also means they're the thing that would catch a
pass↔bridge signature mismatch (see `../../PROJECT_INDEX.md`'s "Pass ↔
bridge coupling" section) that the lit suites structurally cannot, since
those never link against `libc2cudaq.a` at all.

## Files

One pair per pass:

| Pass | Source | Script |
|---|---|---|
| MaxCut | `maxcut_e2e.cpp` | `run_integration.sh` |
| TSP | `tsp_e2e.cpp` | `run_tsp_integration.sh` |
| KColor | `kcolor_e2e.cpp` | `run_kcolor_integration.sh` |
| Clique | `clique_e2e.cpp` | `run_clique_integration.sh` |
| Factor | `factor_e2e.cpp` | `run_factor_integration.sh` |

Each `<name>_e2e.cpp` mirrors the shape its pass matches (see
`PROJECT_INDEX.md`'s pass table) against a small worked example with a
known correct answer, checked in the source itself (prints `PASS`/`FAIL`
and sets exit code accordingly).

`maxcut_actual.cpp` here is a copy of `../maxcut_actual.cpp` (the
reference MaxCut target `test/maxcut/basic.cpp` mirrors) kept alongside
the integration scripts — not itself wired into any `run_*.sh`.

## Running

```bash
cd /home/def3r/def3r/SIGSEGV/LlvmProject/loopanalysis
./test/integration/run_integration.sh          # maxcut
./test/integration/run_tsp_integration.sh
./test/integration/run_kcolor_integration.sh
./test/integration/run_clique_integration.sh
./test/integration/run_factor_integration.sh
```

Each script is self-contained and prints its own 5-step progress (clang
frontend → canonicalize → run the pass → clang backend → link), then runs
the resulting binary. `CLANG`/`OPT`/`CUDAQ_DIR` are overridable via env
vars at the top of each script (no defaults baked in beyond what's
already in the script — same reasoning as `../../tools/README.md`'s
"Environment" section: an absolute path that happens to exist on one
machine isn't safe to assume elsewhere).

Equivalently, once `tools/qoffload-clang++` is set up (see
`../../tools/README.md`), the same pipeline collapses to one command,
e.g.:

```bash
tools/qoffload-clang++ --qpu-pass=factor --qpu-verbose \
  test/integration/factor_e2e.cpp -o factor_bin
```

The `run_*_integration.sh` scripts remain useful on their own for seeing
each pipeline stage's intermediate output explicitly (they don't clean up
their `build/` directory between steps), which is harder to inspect
through the wrapper's single-command interface.

## Prerequisites

Both `build/MinPass.so` (this project) and
`c2cudaq/build/libc2cudaq.a` (the sibling repo, symlinked in as
`../../c2cudaq`) must already be built:

```bash
cd /home/def3r/def3r/SIGSEGV/LlvmProject/loopanalysis
cmake -S . -B build -DLLVM_DIR=/home/def3r/def3r/llvm-project/install/lib/cmake/llvm
cmake --build build

cd /home/def3r/def3r/SIGSEGV/QuantumComp/c2cudaq/build
cmake --build . --target c2cudaq
```
