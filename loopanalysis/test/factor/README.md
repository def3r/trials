# Factor Pass — Test Suite

Tests for the `factor-pass` LLVM out-of-tree plugin (`build/MinPass.so`), matching
the structure and mechanism of the `maxcut-cpp-pass`/`tsp-pass`/`kcolor-pass`/
`clique-pass` suites in `../maxcut/`, `../tsp/`, `../kcolor/`, `../clique/`.

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
cmake --build build --target check-factor
```

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/factor/
```

Run from the `loopanalysis/` root, or from inside `test/factor/` with the full path.

Run a single test verbosely:

```bash
python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/factor/basic_opt.ll -v
```

To see *why* a test does or doesn't fire, run the pass directly with debug
output (requires an assertions-enabled `opt`):

```bash
/home/def3r/def3r/llvm-project/install/bin/opt \
  -load-pass-plugin build/MinPass.so -passes=factor-pass \
  -debug-only=factor-cpp -S test/factor/basic_opt.ll -o /dev/null
```

(`-debug-only=factor-cpp`, not some other pass's debug type — copy-pasting
the wrong one gives total silence even when the pass does attempt a match
and reject. This bit a real debugging session for clique-pass; see
`analysis/clique/clique.md` §1c.)

## Test files

Same two-file convention as `../clique/`, for the same reason: `factor-pass` is a
`FunctionPass` (not a Module pass, unlike kcolor/clique — there's no
self-recursion or call-site hunting here, matching happens directly on the
matched function's own loop nest), and each test's `.cpp` compiles and
canonicalizes cleanly on its own with no `llvm-extract` step needed.

| File | Purpose |
|---|---|
| `<name>.cpp` | Human-readable source. Contains `// CHECK:` directives and a comment explaining exactly which matcher step accepts or rejects it. |
| `<name>_opt.ll` | Optimized IR — the actual lit test input. Has `; RUN:` + `; CHECK:` lines prepended by `update.py`. |

Expected outcome is read directly from the `// CHECK:` / `// CHECK-NOT:` /
`// XFAIL:` directives embedded in the `.cpp`, same mechanism the other four
suites use:

| Directive | Meaning |
|---|---|
| `CHECK: factor_impl` | Should be detected and the loop nest replaced (DETECT). |
| `CHECK-NOT: factor_impl` | Should be correctly rejected. |
| `XFAIL: *` + `CHECK: factor_impl` | Known limitation — currently missed, would fire if fixed. |

## What's being tested — and two surprises found while writing it

Twenty tests, organized by which of `matchFactor()`'s checks they isolate
(see `factor_pass.cpp` and `analysis/factor/factor.md` §1a/§1b for the
full derivation):

- **Signature/scope** (`wrong_arg_count`, `wrong_return_type`, `wide_bounds`):
  the fixed 3-argument signature (`n`, `outA`, `outB`) and `i1`/`i32` type
  requirements.
- **Loop-nesting structure** (`no_inner_loop`, `triple_nested`,
  `loop_starts_elsewhere`): exactly two levels of nesting, entered directly
  from the outer header.
- **Bound/product identity** (`bound_not_direct_arg`, `product_target_not_n`,
  `inner_bound_off_by_one`): the three-times-repeated requirement that the
  outer bound, inner bound, and product compare all resolve to the *same*
  `NArg`.
- **Body shape** (`wrong_mul_operands`, `inverted_predicate`, `non_unit_step`):
  the multiply's operands, the product compare's predicate, and the
  induction step.
- **Fallback shape** (`wrong_fallback_pair`, `store_same_pointer`): the
  "keep it tight" decision that the not-found edge must carry exactly
  `(1, N)`, and that the two outputs go through two genuinely distinct
  pointers.
- **Side-effect gates** (`inner_log`, `exit_log`, `extra_result_use`):
  `checkSideEffects()` (scans the matched loop nest) vs. `matchFactor()`'s
  own step 10 (scans the merge/exit block, which sits *outside* the loop
  and isn't covered by `checkSideEffects()` at all) vs. step 9 (the merge
  phis must have exactly one user, since `performReplacement()` erases
  them outright).
- **Cross-function sanity** (`sibling_function`): an unrelated neighbor
  function in the same file doesn't confuse compilation or matching.
- **Known limitation** (`triangular_start`, XFAIL): `b = a` instead of a
  constant start (skip symmetric duplicate pairs) — a real optimization,
  rejected because the inner phi's preheader value must be a `ConstantInt`.

Two things turned out differently than planned while actually compiling
these, both left documented in the `.cpp` files rather than swept under
the rug:

1. **A true "two different arguments, one per loop bound" false-positive
   probe isn't constructible at all.** The fixed 3-argument signature (one
   `int` slot, two output pointers) that `wrong_arg_count.cpp` enforces
   *also* rules out that entire class of lookalike by construction —
   there's no room for a second bound argument without tripping the arity
   check first. `product_target_not_n.cpp` (comparing against `n+1`
   instead of a second argument) is the closest available probe of the
   same "same `NArg`, checked three times" guard.
2. **instcombine's canonicalization erased more "distinct" shapes than
   expected.** `a*b != n` with inverted branches (originally planned as a
   REJECT) gets canonicalized straight back into the same `icmp eq` +
   normal-branch shape `basic.cpp` produces — reclassified as a DETECT.
   `b <= n` canonicalizes to an inverted `icmp sgt`, not a literal `sle`.
   And `outA = a; outA = b;` (`store_same_pointer.cpp`) only gets
   *partially* dead-store-eliminated, leaving behind an extra basic block
   that trips an earlier structural check than the one it was written to
   test. All three are documented with what's actually observed, not what
   was originally assumed.

## Adding or modifying a test

1. Write (or edit) the `.cpp` file. Add `// CHECK:` / `// CHECK-NOT:` lines at the
   bottom. For a known-miss, add `// XFAIL: *` before the `// CHECK:` line.

   ```cpp
   // --- lit check directives (read by update.py) ---
   // CHECK: replaced factor search loop with call to @factor_impl
   // CHECK: call i1 @factor_impl(i32
   ```

2. Regenerate the `_opt.ll`:

   ```bash
   python3 test/factor/update.py <name>.cpp
   # or regenerate all:
   python3 test/factor/update.py
   ```

3. Before trusting the CHECK directives, run the pass directly with
   `-debug-only=factor-cpp` and inspect the actual `_opt.ll` IR to confirm it
   fails (or succeeds) at the step your test claims to exercise — not some
   earlier, unrelated one. This bit three tests during development here (see
   the "two surprises" section above) — always double check against the
   real compiled IR, not just the source you wrote.

4. Run the tests to confirm:

   ```bash
   python3 /home/def3r/def3r/llvm-project/llvm/utils/lit/lit.py test/factor/
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
