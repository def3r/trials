# MaxCut Pass — Test Suite

Tests for the `maxcut-cpp-pass` LLVM out-of-tree plugin (`build/MinPass.so`).

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
cmake --build build --target check-maxcut
```

- Or with ctest (respects enable_testing()):
```
ctest --test-dir build
   ```

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/maxcut/
```

Run from the `loopanalysis/` root, or from inside `test/maxcut/` with the full path.

Run a single test verbosely:

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/maxcut/tp_basic_opt.ll -v
```

Expected output:

```
-- Testing: 15 tests, 15 workers --
PASS: maxcut-pass :: tp_basic_opt.ll
...
XFAIL: maxcut-pass :: fn_nonzero_init_opt.ll   ← known limitation, tracked by TODO
...
  Passed           : 14
  Expectedly Failed:  1
```

## Test files

Each test case has three files:

| File | Purpose |
|---|---|
| `<name>.cpp` | Human-readable source. Contains `// CHECK:` directives. |
| `<name>_opt.ll` | Optimized IR — the actual lit test input. Has `; RUN:` + `; CHECK:` lines prepended by `update.py`. |
| `<name>_ex.ll`, `<name>.ll` | Intermediate build artefacts. Not run by lit. |

## Naming convention

| Prefix | Meaning |
|---|---|
| `tp_` | True Positive — should be detected and replaced |
| `tn_` | True Negative — should not be detected |
| `fp_` | False Positive that the pass now correctly rejects |
| `fn_` | False Negative — known limitation or previously missed, now fixed |

## Adding or modifying a test

1. Write (or edit) the `.cpp` file. Add `// CHECK:` / `// CHECK-NOT:` lines at the
   bottom. For a known-miss, add `// XFAIL: *` before the `// CHECK:` line.

   ```cpp
   // --- lit check directives (read by update.py) ---
   // CHECK: replaced MaxCut loops with call to @maxcut_impl
   // CHECK: call i32 @maxcut_impl(ptr
   ```

2. Register the function grep pattern in `update.py`'s `FUNC_GREP` dict if it is a
   new file (maps the `.cpp` basename to a regex that identifies the target function's
   mangled name).

3. Regenerate the `_opt.ll`:

   ```bash
   python3 test/maxcut/update.py <name>.cpp
   # or regenerate all:
   python3 test/maxcut/update.py
   ```

4. Run the tests to confirm:

   ```bash
   python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/maxcut/
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
