# KColor Pass — Test Suite

Tests for the `kcolor-pass` LLVM out-of-tree plugin (`build/MinPass.so`), matching
the structure and mechanism of the `maxcut-cpp-pass`/`tsp-pass` suites in
`../maxcut/` and `../tsp/`.

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
cmake --build build --target check-kcolor
```

- Or with ctest (respects enable_testing()):
```
ctest --test-dir build
```

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/kcolor/
```

Run from the `loopanalysis/` root, or from inside `test/kcolor/` with the full path.

Run a single test verbosely:

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/kcolor/basic_opt.ll -v
```

To see *why* a test does or doesn't fire, run the pass directly with debug
output (requires an assertions-enabled `opt`):

```bash
/home/def3r/def3r/llvm-project/install/bin/opt \
  -load-pass-plugin build/MinPass.so -passes=kcolor-pass \
  -debug-only=kcolor-cpp -S test/kcolor/basic_opt.ll -o /dev/null
```

## Test files

Each test case has three files:

| File | Purpose |
|---|---|
| `<name>.cpp` | Human-readable source. Contains `// CHECK:` directives and a comment explaining exactly which matcher step accepts or rejects it. |
| `<name>_opt.ll` | Optimized IR — the actual lit test input. Has `; RUN:` + `; CHECK:` lines prepended by `update.py`. |
| `<name>_ex.ll`, `<name>.ll` | Intermediate build artefacts. Not run by lit. |

Expected outcome is read directly from the `// CHECK:` / `// CHECK-NOT:` /
`// XFAIL:` directives embedded in the `.cpp`, same mechanism `update.py`
uses for MaxCut/TSP:

| Directive | Meaning |
|---|---|
| `CHECK: kcolor_impl` | Should be detected and a top-level call site replaced (DETECT). |
| `CHECK-NOT: kcolor_impl` | Should be correctly rejected, or matched-but-nothing-to-replace (REJECT). |
| `XFAIL: *` + `CHECK: kcolor_impl` | Known limitation — currently missed, would fire if fixed. |

## Why this test suite looks different from MaxCut/TSP's

MaxCut and TSP both solve their problem with an **outer loop enumerating an
exponential candidate space** (all subsets, all permutations) — everything
that pattern needs to check lives inside `LoopInfo` for a single function.
KColor's m-coloring decision problem is solved by **self-recursion**
instead: `solve(node, ...)` tries each color at `node` and recurses on
`node+1`. There's no outer loop to replace, so:

- There are no `open_path`/`wrong_close_index`-style tests here (no
  wrap-around epilogue to have).
- There are no `min_cmp_form`/`scaled_cost`-style tests here (no
  accumulator — the return value is a bare `bool`, nothing to scale or
  compare against a running best).
- Instead, most of this suite targets `kcolor_pass.cpp`'s `matchSolve()`
  steps directly: exactly one self-call, the `node+1` recursion argument,
  every other argument threaded through unchanged, the base-case compare,
  the guard call, and the assign/backtrack store pair on the same
  container.
- The transform is **inter-procedural**: `solve()`'s shape is matched in
  one function, but the call site actually replaced (`solve(0, ...)`) may
  live in a different one (`graphColoring`, or several). `kcolor-pass` is
  a **Module** pass, not a Function pass, and several tests
  (`no_top_level_call`, `multi_call_site`, `plain_call_site`) exist
  specifically to exercise that split.

## Extracting multiple functions per test

Unlike `../maxcut/`/`../tsp/`'s `update.py` (single `-func=<mangled name>`
per test), a kcolor test needs the guard function, the recursive solve
function, and at least one top-level call site all kept as real
definitions — `llvm-extract`'s `-func=` only keeps the exact name(s)
given, so a single-function extraction would leave everything else as a
declaration and the match would never see anything to work with.

Every test's functions share a name prefix (e.g. `kc_basic_isSafe`,
`kc_basic_solve`, `kc_basic_graphColoring`), and `update.py` passes that
prefix straight to `llvm-extract -rfunc=<regex>`, which keeps every
matching function as a definition in one pass. When adding a new test,
give its functions a fresh, unique prefix and register `{name: prefix}` in
`update.py`'s `FUNC_GREP` dict.

## A structural quirk worth knowing before adding a new test

`matchSolve()` currently requires the solve-shaped function to take
**exactly 5 arguments** (node, color, m, N, graph) and **return `i1`**.
These are deliberate scope boundaries, not limitations pending a fix —
see `wrong_arg_count.cpp` and `int_return.cpp`, which document them as
such (correctly rejected, not XFAIL). If you write a test whose solve
function has a different arity or return type for some other reason,
expect it to be rejected at that check specifically, and say so in the
test's comment rather than assuming a different step caught it — this
bit several tests in the TSP suite (see `../tsp/README.md`'s note on
`volatile` accumulators) and is worth double-checking with
`-debug-only=kcolor-cpp` before finalizing `// CHECK:` lines.

## Adding or modifying a test

1. Write (or edit) the `.cpp` file, using a unique function-name prefix.
   Add `// CHECK:` / `// CHECK-NOT:` lines at the bottom. For a known-miss,
   add `// XFAIL: *` before the `// CHECK:` line.

   ```cpp
   // --- lit check directives (read by update.py) ---
   // CHECK: replaced top-level solve() call with call to @kcolor_impl
   // CHECK: call i1 @kcolor_impl(ptr
   ```

2. Register `{name: prefix}` in `update.py`'s `FUNC_GREP` dict.

3. Regenerate the `_opt.ll`:

   ```bash
   python3 test/kcolor/update.py <name>.cpp
   # or regenerate all:
   python3 test/kcolor/update.py
   ```

4. Before trusting the CHECK directives, run the pass directly with
   `-debug-only=kcolor-cpp` and confirm it fails (or succeeds) at the step
   your test claims to exercise — not some earlier, unrelated one.

5. Run the tests to confirm:

   ```bash
   python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/kcolor/
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
.cpp  ──update.py──►  .ll  ──llvm-extract -rfunc──►  _ex.ll  ──opt (passes)──►  _opt.ll
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
