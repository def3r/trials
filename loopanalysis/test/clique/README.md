# Clique Pass — Test Suite

Tests for the `clique-pass` LLVM out-of-tree plugin (`build/MinPass.so`), matching
the structure and mechanism of the `maxcut-cpp-pass`/`tsp-pass`/`kcolor-pass`
suites in `../maxcut/`, `../tsp/`, `../kcolor/`.

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
cmake --build build --target check-clique
```

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/clique/
```

Run from the `loopanalysis/` root, or from inside `test/clique/` with the full path.

Run a single test verbosely:

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/clique/basic_opt.ll -v
```

To see *why* a test does or doesn't fire, run the pass directly with debug
output (requires an assertions-enabled `opt`):

```bash
/home/def3r/def3r/llvm-project/install/bin/opt \
  -load-pass-plugin build/MinPass.so -passes=clique-pass \
  -debug-only=clique-cpp -S test/clique/basic_opt.ll -o /dev/null
```

(`-debug-only=clique-cpp`, not some other pass's debug type — copy-pasting
the wrong one gives total silence even when the pass does attempt a match
and reject at a gate. This bit a real debugging session; see
`analysis/clique/clique.md` §1c.)

## Test files — and a difference from maxcut/tsp/kcolor

Each test case has **two** files here, not three:

| File | Purpose |
|---|---|
| `<name>.cpp` | Human-readable source. Contains `// CHECK:` directives and a comment explaining exactly which matcher step accepts or rejects it. |
| `<name>_opt.ll` | Optimized IR — the actual lit test input. Has `; RUN:` + `; CHECK:` lines prepended by `update.py`. |

There's no `<name>_ex.ll` extracted-function file. `clique-pass` is a
**Module** pass and scans every function in the file regardless of which
one holds the pattern, so `update.py` compiles and canonicalizes the whole
translation unit directly — no `llvm-extract` step, and no need to register
a function-name pattern per test the way `../kcolor/update.py`'s
`FUNC_GREP` does. One consequence: `instcombine` needs
`<no-verify-fixpoint>` in the pipeline, since without it instcombine hits a
hard error on one of the `std::vector` template instantiations elsewhere in
the file — code the pass never looks at, but still has to survive
canonicalization to produce a `_opt.ll` at all.

Expected outcome is read directly from the `// CHECK:` / `// CHECK-NOT:` /
`// XFAIL:` directives embedded in the `.cpp`, same mechanism the other
three suites use:

| Directive | Meaning |
|---|---|
| `CHECK: clique_impl` | Should be detected and a top-level call site replaced (DETECT). |
| `CHECK-NOT: clique_impl` | Should be correctly rejected, or matched-but-nothing-to-replace (REJECT). |
| `XFAIL: *` + `CHECK: clique_impl` | Known limitation — currently missed, would fire if fixed. |

## Why this suite looks different from kcolor's

`maxCliques`'s self-recursion shape has more moving parts than kcolor's
`solve()` (see `analysis/clique/clique.md` §1b for the full IR-verified
derivation), so several checks here don't have a kcolor analog at all:

- **Two independent "+1" sources in one self-call**, not one. The `start`
  argument advances via the candidate loop's own induction variable
  (`v + 1`); the `size` argument advances via a formal parameter
  (`size + 1`, kcolor's style). Getting either one wrong on its own is a
  distinct test (`non_unit_start` / `non_unit_size`), since they're
  checked independently.
- **A "loop starts from `start`" check with no kcolor equivalent**
  (`loop_starts_elsewhere`) — kcolor's candidate loop's bound was a plain,
  search-independent value; clique's loop must additionally begin from the
  function's own `start` parameter, which is a fact worth confirming, not
  assuming.
- **The assign store's stored value is checked, not just its index**
  (`assign_wrong_value`) — kcolor never checked what value got written
  into `color[node]`, only where. `clique[size]` must specifically hold
  the loop's own candidate vertex.
- **A running-best accumulator updated twice per candidate, never an
  early exit** (`accumulator_container_mismatch`, `missing_accept_update`,
  `extra_max_call`, `extend_max_wrong_operand`) — structurally closer to
  `maxcut_pass.cpp`'s running-best detection than to kcolor's
  early-return-on-first-success shape, so this whole category is new here.
- **`accumulator_promoted_to_phi` is XFAIL, not REJECT** — mirrors TSP's
  `min_cmp_form.cpp`. `best` only stays memory-resident because
  `std::max<int>` takes its arguments by reference; a hand-written
  comparison removes the only thing forcing that, `mem2reg` promotes
  `best` straight to an SSA phi, and the matcher never sees it. Documented
  as a known limitation, not solved.
- **`longest_path_lookalike` is a deliberate false-positive probe**, same
  spirit as kcolor's `nqueens_lookalike.cpp`: a different problem (longest
  simple path, single-edge guard instead of all-pairs clique check)
  reshaped into the same self-call/assign/running-max pattern. Whether it
  matches is reported as found, not assumed either way.

## Adding or modifying a test

1. Write (or edit) the `.cpp` file. Add `// CHECK:` / `// CHECK-NOT:` lines at the
   bottom. For a known-miss, add `// XFAIL: *` before the `// CHECK:` line.

   ```cpp
   // --- lit check directives (read by update.py) ---
   // CHECK: replaced top-level maxCliques() call with call to @clique_impl
   // CHECK: call i32 @clique_impl(ptr
   ```

2. Regenerate the `_opt.ll`:

   ```bash
   python3 test/clique/update.py <name>.cpp
   # or regenerate all:
   python3 test/clique/update.py
   ```

3. Before trusting the CHECK directives, run the pass directly with
   `-debug-only=clique-cpp` and confirm it fails (or succeeds) at the step
   your test claims to exercise — not some earlier, unrelated one. This
   bit several tests in both the TSP and kcolor suites during development
   (a test meant to exercise one specific check instead tripped an earlier,
   unrelated one, like an arity mismatch) — always double check.

4. Run the tests to confirm:

   ```bash
   python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/clique/
   ```

## Environment overrides

`update.py` respects these environment variables:

| Variable | Default |
|---|---|
| `CLANG` | `clang` |
| `OPT` | `/home/def3r/def3r/llvm-project/install/bin/opt` |

`lit.cfg.py` respects:

| Variable | Default |
|---|---|
| `LLVM_PROJECT_INSTALL` | `/home/def3r/def3r/llvm-project/install` |
| `LLVM_PROJECT_BUILD` | `/home/def3r/def3r/llvm-project/build` |

## How it works

```
.cpp  ──update.py──►  .ll  ──opt (passes)──►  _opt.ll
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
