# TSP Pass — Test Suite

Tests for the `tsp-pass` LLVM out-of-tree plugin (`build/MinPass.so`), matching
the structure and mechanism of the `maxcut-cpp-pass` suite in `../maxcut/`.

## Prerequisites

Build the plugin first:

```bash
cd /home/def3r/def3r/SIGSEGV/LlvmProject/loopanalysis
cmake -S . -B build -DLLVM_DIR=/home/def3r/def3r/llvm-project/install/lib/cmake/llvm
cmake --build build
```

## Running the tests

- Build plugin and run all lit tests:
```
cmake --build build --target check-tsp
```

- Or with ctest (respects enable_testing()):
```
ctest --test-dir build
```

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/tsp/
```

Run from the `loopanalysis/` root, or from inside `test/tsp/` with the full path.

Run a single test verbosely:

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/tsp/basic_opt.ll -v
```

To see *why* a test does or doesn't fire, run the pass directly with debug
output (requires an assertions-enabled `opt`):

```bash
/home/def3r/def3r/llvm-project/install/bin/opt \
  -load-pass-plugin build/MinPass.so -passes=tsp-pass \
  -debug-only=tsp-cpp -disable-output test/tsp/basic_opt.ll
```

## Test files

Each test case has three files:

| File | Purpose |
|---|---|
| `<name>.cpp` | Human-readable source. Contains `// CHECK:` directives and a comment explaining exactly which matcher step accepts or rejects it. |
| `<name>_opt.ll` | Optimized IR — the actual lit test input. Has `; RUN:` + `; CHECK:` lines prepended by `update.py`. |
| `<name>_ex.ll`, `<name>.ll` | Intermediate build artefacts. Not run by lit. |

## Naming and outcomes

Unlike `../maxcut/`'s original `tp_`/`tn_`/`fp_`/`fn_` prefix convention,
filenames here are plain descriptive names; the expected outcome is read
directly from the `// CHECK:` / `// CHECK-NOT:` / `// XFAIL:` directives
embedded in the `.cpp` (same mechanism `update.py` and `gen_ir.sh` use for
MaxCut):

| Directive | Meaning |
|---|---|
| `CHECK: replaced TSP loop...` | Should be detected and replaced (DETECT). |
| `CHECK-NOT: tsp_impl` | Should be correctly rejected (REJECT). |
| `XFAIL: *` + `CHECK: tsp_impl` | Known limitation — currently missed, would fire if fixed. |

## A structural quirk specific to TSP: accumulators must stay memory-resident

tsp_pass.cpp's matchers work on a load-add-store shape (`AllocaInst` +
`LoadInst`/`StoreInst`), *not* phi backedges — see the header comment in
`tsp_pass.cpp`. That shape only survives `sroa`+`mem2reg` because
`std::min<int>(const int&, const int&)` takes its arguments by reference,
forcing `currCost`/`minCost`'s addresses to escape.

**If a test's source doesn't call `std::min`/`std::max` (or otherwise take
the accumulator's address), mem2reg promotes it straight to an SSA phi, and
the inner scoring loop won't match at all** — regardless of what the test
is actually trying to exercise. Several tests in this suite
(`sle_compare.cpp`, `sum_cost.cpp`, `score_no_min.cpp`) declare their
accumulators `volatile` specifically to route around this and isolate the
check they're actually targeting; `min_cmp_form.cpp` documents the
promotion itself as the (currently unfixed) limitation. See the comment at
the top of any of those files for the full explanation before adding a new
test that uses a hand-written comparison instead of `std::min`.

## Adding or modifying a test

1. Write (or edit) the `.cpp` file. Add `// CHECK:` / `// CHECK-NOT:` lines at the
   bottom. For a known-miss, add `// XFAIL: *` before the `// CHECK:` line.

   ```cpp
   // --- lit check directives (read by update.py) ---
   // CHECK: replaced TSP loop with call to @tsp_impl
   // CHECK: call i32 @tsp_impl(ptr
   ```

2. Register the function grep pattern in `update.py`'s `FUNC_GREP` dict (maps the
   `.cpp` basename to a regex that identifies the target function's mangled name).

3. Regenerate the `_opt.ll`:

   ```bash
   python3 test/tsp/update.py <name>.cpp
   # or regenerate all:
   python3 test/tsp/update.py
   ```

4. Before trusting the CHECK directives, run the pass directly with
   `-debug-only=tsp-cpp` (see above) and confirm it fails (or succeeds) at
   the step your test claims to exercise — not some earlier, unrelated step
   (see the memory-residency quirk above; this bit several tests in this
   suite during development).

5. Run the tests to confirm:

   ```bash
   python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/tsp/
   ```

## Environment overrides

`update.py` respects these environment variables:

| Variable | Default |
|---|---|
| `CLANG` | `clang` |
| `OPT` | `/home/def3r/def3r/llvm-project/install/bin/opt` |
| `LLVM_EXTRACT` | `llvm-extract` |

`lit.cfg.py` respects:

| Variable | Default |
|---|---|
| `LLVM_PROJECT_INSTALL` | `/home/def3r/def3r/llvm-project/install` |
| `LLVM_PROJECT_BUILD` | `/home/def3r/def3r/llvm-project/build` |

## How it works

```
.cpp  ──update.py──►  .ll  ──opt -O0──►  _ex.ll  ──opt (passes)──►  _opt.ll
                                                                         │
                                                        RUN: + CHECK: prepended
                                                                         │
                                                 lit ──► opt (pass) ──► FileCheck
```

- `update.py` runs the full compilation pipeline and embeds the `// CHECK:` lines
  from the `.cpp` as `; CHECK:` directives in the `_opt.ll`.
- `lit` reads the `; RUN:` line in each `_opt.ll`, executes the pass, and pipes
  the output through `FileCheck`, which matches against the `; CHECK:` lines in
  the same file.
