# Project Index — LLVM QPU-Offload Passes

Source-to-source index for generating a report on this project, and the
first thing to read if you're a fresh agent picking this project up cold.
All paths are relative to `loopanalysis/` (this file's directory) unless
marked otherwise. Compiled and verified by walking the actual filesystem
and grepping actual source — where something looked undocumented or a
claim needed checking, it was checked, not assumed.

## What this project is

Five LLVM passes, each built as an out-of-tree plugin (`build/MinPass.so`),
that recognize a specific brute-force/backtracking classical algorithm
shape in compiled C++ and replace the matched IR with a call to a
"bridge" function. Each bridge lives in `c2cudaq/src/bridge.cpp` (a
symlink to `/home/def3r/def3r/SIGSEGV/QuantumComp/c2cudaq`) and either
calls a quantum kernel (with a classical fallback) or runs classical-only,
depending on what that kernel's own reliability/performance evaluation
found.

## The five passes — what each one identifies

| Pass | Source | Registered as | Pass-manager kind | What it matches |
|---|---|---|---|---|
| MaxCut | `maxcut_pass.cpp` | `maxcut-pass` | Function | A cut-edge-counting loop over a partition/edge list, accumulating crossing-edge count. |
| TSP | `tsp_pass.cpp` | `tsp-pass` | Function | Brute-force permutation search: inner loop scores a path via a cost matrix, outer loop walks `std::next_permutation`, tracks min cost via `std::min`. |
| KColor | `kcolor_pass.cpp` | `kcolor-pass` | Module | Self-recursive backtracking m-coloring (`isSafe`/`solve`/`graphColoring` shape), early-return on first success. |
| Clique | `clique_pass.cpp` | `clique-pass` | Module | Self-recursive backtracking max-clique search (`isClique`/`maxCliques`/`findMaxClique` shape), running-best via two `std::max` calls, no early exit, no backtrack store. |
| Factor | `factor_pass.cpp` | `factor-pass` | Function | Brute-force nested-loop search over pairs `(a, b)` for `a*b == n`, tight `(1, N)` fallback on exhaustion. |

Registration hub: `pass.cpp` — declares and calls all five
`register*Pass()` functions. Nothing else; the legacy demo matcher this
file used to also carry (a `min-pass` cut-edge-counting prototype,
superseded by the real `maxcut-pass`) has been removed — it was dead
code, referenced by nothing (`grep -rn "min-pass"` across the whole
project matched only its own definition before removal), and its
presence in the registration hub with zero explanation was itself a
standing documentation gap. Confirmed after removal: `MinPass.so`
rebuilds clean, and all five `check-<name>` lit targets plus all five
`test/integration/run_*.sh` scripts still pass.

## Pass ↔ bridge coupling — read this before touching either side

This is the single fact most likely to cause a silent bug if missed, and
it isn't enforced by any compiler check, shared header, or test that
runs by default.

**The mechanism**: each pass's `performReplacement()` builds an LLVM
`FunctionType` *by hand* — argument count, each argument's exact type
(`i32` vs `i64`, `ptr` vs a scalar, order), and the return type — then
emits a call against it via `Mod->getOrInsertFunction("<name>_impl",
FTy)`. That declaration is pure assertion: at `opt` time, `bridge.cpp`
hasn't been compiled into the module yet (it's a separate translation
unit, only linked in at the final `clang` step), so nothing checks the
assumed `FunctionType` against the real one. If they disagree, you get
either a linker-level mismatch or — worse — a *silent* ABI mismatch
(wrong values passed in the wrong registers, garbage results, no error
at all). This exact bug already happened once, for real: kcolor's
`wide_bounds` case, where `performReplacement` assumed `i32` without the
matcher first checking the source's actual argument width. That incident
is why `clique_pass.cpp` and `factor_pass.cpp` both explicitly gate their
match on `NArg->getType()->isIntegerTy(32)` — a matcher-side mitigation
(decline to fire rather than emit a wrong call), not a fix for the
coupling itself.

**Consequence**: if you change a bridge function's C++ signature in
`bridge.cpp` (add a parameter, widen an `int` to `int64_t`, reorder
pointers, change the return type), you must *manually* update the
matching `FunctionType::get(...)` call in that pass's `performReplacement`
to match — nothing will tell you if you forget except a crash or wrong
answer at `test/integration/run_*.sh` runtime (see that directory's
README for why the lit suites specifically *can't* catch this: they
never link against `libc2cudaq.a` at all).

**Verified-current signatures** (both sides read directly from source,
cross-checked against each other — all five agree as of this writing):

| Pass | Emitted call (`performReplacement`, `<pass>.cpp`) | Real definition (`bridge.cpp`) |
|---|---|---|
| MaxCut | `i32 @maxcut_impl(ptr, ptr, ptr)` | `int maxcut_impl(const vector<vector<int>>*, const vector<pair<int,int>>*, vector<int>*)` |
| TSP | `i32 @tsp_impl(ptr, ptr)` | `int tsp_impl(const vector<int>*, const vector<vector<int>>*)` |
| KColor | `i1 @kcolor_impl(ptr, i32, i32, ptr)` | `bool kcolor_impl(const vector<vector<int>>*, int m, int N, vector<int>*)` |
| Clique | `i32 @clique_impl(ptr, i32, ptr)` | `int clique_impl(const vector<vector<int>>*, int N, vector<int>*)` |
| Factor | `i1 @factor_impl(i32, ptr, ptr)` | `bool factor_impl(int n, int* outA, int* outB)` |

If you're auditing or extending a pass: re-run this cross-check
(`grep -n "FunctionType::get\|getOrInsertFunction" *_pass.cpp` against
the real function signature in `c2cudaq/src/bridge.cpp`) before trusting
this table — it reflects one point in time, not a guarantee.

## Where the tests live, and how they work

Each pass has its own lit-based test suite under `test/<name>/`:

| Suite | README | Test count / shape |
|---|---|---|
| `test/maxcut/` | `test/maxcut/README.md` | — |
| `test/tsp/` | `test/tsp/README.md` | — |
| `test/kcolor/` | `test/kcolor/README.md` | — |
| `test/clique/` | `test/clique/README.md` | 24 tests (22 REJECT/DETECT + 2 XFAIL) |
| `test/factor/` | `test/factor/README.md` | 20 tests (19 REJECT/DETECT + 1 XFAIL) |

Mechanism, common to all five: a `<name>.cpp` source file with `// CHECK:`
/ `// CHECK-NOT:` / `// XFAIL:` comments; `test/<name>/update.py` compiles
it through that pass's specific canonicalization pipeline and embeds the
CHECK directives as `; CHECK:` lines in a generated `<name>_opt.ll`; `lit`
runs the pass against that file and pipes output through `FileCheck`.
`CMakeLists.txt` exposes one `check-<name>` target per suite.

**Two different canonicalization pipelines are in use, and mixing them up
silently breaks matching** — the single most load-bearing fact about this
whole project:

- **Loop family** (maxcut, tsp): needs `loop-simplify,lcssa,indvars` in
  the pipeline.
- **Reduced family** (kcolor, clique, factor): actively *broken* by
  `indvars` — it buries the loop-bound argument inside an `llvm.smax`
  call, which the `NArg` check in each of those three matchers can't see
  through.

(Documented independently in `analysis/clique/clique.md` §1c,
`analysis/factor/factor.md` §1c, and `tools/README.md`; stated once here
as the canonical version.)

End-to-end integration tests (compile → pass → link against
`libc2cudaq.a` + the cudaq runtime → run → check the actual numeric
answer) live in `test/integration/` — see `test/integration/README.md`
(new; this project had no doc for that directory before).

## Where the kernel/algorithm analysis lives

All five quantum kernels backing these passes have now been
stress-tested for correctness and performance. KColor is the one
exception with no standalone status doc (see "Open, deliberately
unaddressed items" below — a deliberate scope call, not an oversight):

| Kernel | Analysis dir | Status doc | Stress-test report |
|---|---|---|---|
| `c2q_maxcut` (QAOA) | `analysis/maxcut/` | `analysis/maxcut/maxcut.md` | `analysis/maxcut/report.html` |
| `c2q_tsp` (QAOA) | `analysis/tsp/` | `analysis/tsp/tsp.md` | `analysis/tsp/report.html` |
| `c2q_kcolor` (QAOA) | `analysis/kcolor/` | *none (deliberately not written — see below)* | `analysis/kcolor/report.html` |
| `c2q_clique` (QAOA) | `analysis/clique/` | `analysis/clique/clique.md` | `analysis/clique/report.html` |
| `c2q_factor` (Grover) | `analysis/factor/` | `analysis/factor/factor.md` | published Artifact (linked from `factor.md` §2) |

`clique.md`, `factor.md`, `maxcut.md`, and `tsp.md` are the four "status
notes" documents in this project — living docs covering: which LLVM
passes are needed and why (IR-verified, not just reasoned about — for
clique/factor; maxcut/tsp's passes predate this convention), the kernel
stress-test findings and a practical-limits table, and a next-steps list.

**Headline results, since they differ sharply by kernel:**
- **MaxCut**: no invalid-answer mode at all (every bitstring is a valid
  cut) — completeness depends almost entirely on graph *regularity*, not
  qubit count (ring graphs: 100% at every size tested; random sparse
  graphs: 0% by N=14). ~3× cheaper wall-clock than clique at matched
  qubit count.
- **TSP**: the one kernel that scales as **N² qubits, not N** — every
  other kernel here is linear in node count. Crosses the 28-qubit
  simulator ceiling at just 6 cities, and completeness had already
  collapsed to 0/10 by N=4 (16 qubits). A stress-test run on this
  project's own hardware couldn't even *complete* a single N=5 (25-qubit)
  seed within 20 minutes at 100% GPU utilization — killed rather than
  pushed further. **The severity of this collapse turned out to have a
  real cause worth checking**: `c2cudaq`'s `qubo_tsp` (in
  `c2cudaq/include/c2cudaq/internal.h`) had two genuine bugs — it never
  scored the tour's closing edge, and (more severe) about half its
  distance-cost terms were silently written into a matrix triangle
  nothing downstream ever reads. Both confirmed via pure-classical
  brute-force verification (no GPU needed), both fixed, and the fix
  verified: the corrected QUBO's own true minimum now exactly matches
  the classical optimum. **Re-running the empirical QAOA test after the
  fix showed no improvement** — the completeness collapse is a genuine
  QAOA convergence difficulty, not a masked implementation bug. Net
  result: `tsp_impl`'s classical-only choice is confirmed even more
  strongly than before, and `qubo_tsp` itself is now correct for any
  future caller (this fix isn't TSP-pass-specific — it lives in the
  shared c2cudaq library). See `analysis/tsp/tsp.md` §1a for the full
  bug writeup and `analysis/tsp/tsp_qubo_check.cpp` for the verifier.
- **KColor / Clique**: sound but incomplete, wall-clock cost turns
  exponential well inside the 28-qubit ceiling. See their own `.md` docs.
- **Factor**: sound *and* complete everywhere tested (unique among the
  five), but the entire usable range (n≤127) is one classical trial
  division clears in nanoseconds — 5–10 orders of magnitude faster than
  the kernel. `factor_pass.cpp` was still built, deliberately, for when
  non-simulated hardware changes that calculus.

`c2cudaq/README.md` (the sibling repo's own root, not this project)
documents the kernel *API* itself (function signatures, QAOA vs. VQE,
factoring approach) — what each `c2q_*` function does, separate from how
well it performs. `c2cudaq/claude.md` is a design doc for a *different,
unrelated, in-progress* feature (SQIF factorization) — don't confuse it
for general project documentation.

**A correction that touches every report above**: all of them were
generated on GPU (`nvidia` target, cuStateVec FP32), not CPU (`qpp-cpu`,
FP64) as the clique/factor reports originally claimed in their footers.
`nvq++` silently defaults to the GPU-backed `nvidia` target whenever a
CUDA GPU is present on the compiling machine — confirmed directly in its
own script logic, not assumed. Full story and the fix in
`analysis/tsp/tsp.md` §2; all affected footers have been corrected.

## The classical/quantum bridge layer

`c2cudaq/src/bridge.cpp` — one translation unit, one `extern "C"`
function per pass. Each has a doc comment immediately above it stating
the exact call signature the corresponding pass emits (see the coupling
table above) and why it is or isn't kernel-backed.

**Project-wide policy, as of this writing**: kernel-first wherever a
kernel has been verified correct within some range, classical as the
fallback (out-of-range input, or an ambiguous kernel "no"/negative
answer) — never the default. Wall-clock cost lost its veto: `factor_impl`
was deliberately switched from "not worth it, too slow" to kernel-first
specifically to exercise the quantum path within its verified-correct
range regardless of speed. `clique_impl` is the one holdout, and for a
different, harder reason than speed (see below).

- `maxcut_impl` — kernel-first (`c2q_maxcut`, QAOA) **plus an exact
  classical comparison below N≤16** (cheap exhaustive enumeration),
  always keeping the better cut. Added because the kernel has no
  invalid-answer mode at all — every bitstring is a valid cut — so
  there's no decode-time signal distinguishing "found the true optimum"
  from "settled for a valid-but-suboptimal one." Above N=16 the bridge
  stays kernel-only, a documented remaining risk, not an oversight.
- `kcolor_impl` — kernel-first with **mandatory** classical fallback
  (`c2q_kcolor` is sound but not complete; hit-rate numbers and the qubit
  safety cutoff `kKColorQubitSafeLimit = 24` are in the comment itself).
  Unchanged this session — the original template the other three
  kernel-first bridges now follow.
- `tsp_impl` — kernel-first **only at total city count ≤3**. Narrower
  than it looks on purpose: `decode_tsp` only proves a returned tour is
  *valid*, not that it's the *minimum-cost* one, and at N≤3 a complete
  graph has exactly one distinct Hamiltonian cycle, so valid and optimal
  provably coincide there — a degenerate property that stops holding at
  N=4 and must not be assumed at a higher cutoff without re-deriving it.
  Classical fallback for N>3 or an ambiguous kernel "no valid tour"
  answer (0% success was already observed at N=4 in the stress test).
- `clique_impl` — **still classical-only**, and unlike the other three,
  this one wasn't just "not gotten to yet": applying the same pattern
  here safely is harder. `decode_clique` also only proves *validity*
  (a genuine clique), not *maximality* — and unlike TSP, clique has no
  small-N degenerate case where validity trivially implies maximality
  (even a 3-vertex graph can have cliques of size 0–3, so a valid answer
  there doesn't pin down the true max the way a valid TSP tour does at
  N≤3). Doing this correctly would mean either accepting a weaker
  "best-found, not provably maximum" contract in some size band, or a
  genuinely more complex hybrid (kernel result as a lower-bound hint,
  classical branch-and-bound to close the gap) — a real design decision,
  not a mechanical port of the other three, deliberately left open rather
  than guessed at.
- `factor_impl` — kernel-first for `4 ≤ n ≤ 127` (the kernel's entire
  verified-correct range), classical fallback outside it or on an
  ambiguous "no factor found." A positive kernel answer is self-verified
  for free (`a*b==n`) before being trusted. See
  `analysis/factor/factor.md`'s updated verdict for the full reasoning —
  correctness was never in question here, this was purely a scope
  decision about whether wall-clock cost should gate kernel use at all.

**A second, unrelated bug found while re-verifying these changes**: every
`test/integration/run_*_integration.sh` script and `tools/qoffload-clang++`
hardcoded the CPU backend (`-lnvqir-qpp`) at their link step, regardless
of whether a GPU was present — unlike raw `nvq++`, which auto-detects and
prefers GPU. This meant the actual pass-generated pipeline could never
reproduce the GPU wall-clock numbers this project's own `.md` reports
document, no matter what hardware ran it. Confirmed and fixed: all six
scripts now replicate `nvq++`'s own `query_gpu()` check. See
`analysis/tsp/tsp.md`'s next-steps item 5 for the full story (found while
re-testing `tsp_impl`'s kernel-first change, not TSP-specific — it
affected every bridge equally).

## The single-command wrapper

`tools/qoffload-clang++` + `tools/README.md` — collapses the
clang → opt → clang → link pipeline into one command, gated behind
`--qpu-pass=<name>` (opt-in only). Documents the loop/reduced pipeline
split and the Module-pass-vs-Function-pass distinction that made a naive
merged `-passes=` string fail during development. Every env var it needs
(`OPT`, `LLVM_LINK`, `CUDAQ_DIR`, `C2CUDAQ_ROOT`, `QOFFLOAD_PASS`) is
required with no machine-specific default except `CLANG`/`OPT`
(`clang++`/`opt` resolved via `PATH` — portable by construction, unlike
an absolute path).

## `test/` — what's important, what to ignore

`test/` has ~40 loose files at its root alongside the real suite
directories. **Important, load-bearing files:**

- `test/maxcut.cpp`, `test/maxcut_actual.cpp`, `test/tsp.cpp`,
  `test/kcolor.cpp`, `test/clique.cpp`, `test/factor.cpp` — the reference
  source each pass's test suite mirrors (e.g. `test/clique/basic.cpp`'s
  header comment says exactly this). `test/maxcut_actual.cpp` in
  particular is what `test/maxcut/basic.cpp` was built against, distinct
  from `test/maxcut.cpp`'s own `actual()`/`actual2()` functions — both
  are referenced, not a stray duplicate.
- `test/maxcut/`, `test/tsp/`, `test/kcolor/`, `test/clique/`,
  `test/factor/` — the lit suites.
- `test/integration/` — the e2e scripts (see its own README).

**Everything else directly under `test/`** — the various `clean*.ll`,
`clean*.png`, `*.dot`, `mc.c`/`mc.ll`, `maxcutV2.ll`, `maxcut_actual_*.
{ll,s}`, `cfg.png`, `actual.png`, a stray `a.out`, and similar — is
leftover intermediate/experimentation output. Not referenced by any test
runner, `update.py`, or doc in this project. Safe to ignore when reading
the project; not evaluated here for whether it's safe to *delete* (that's
a separate call the project owner should make, not inferred from absence
of references alone). `test/viz/` is an empty directory with no stated
purpose.

## Open, deliberately unaddressed items

Not gaps to fill blindly — each has a reason it's being left as-is for
now, stated so a future pass at this doesn't have to rediscover why:

1. **KColor has no standalone `kcolor.md`, and this pass is not writing
   one.** `clique.md`/`factor.md`/`maxcut.md`/`tsp.md`'s format would be
   the template if someone does this later — `analysis/kcolor/report.html`
   and the dense summary in `bridge.cpp`'s `kcolor_impl` comment are what
   exists today.
2. **`loopanalysis.md` (root, 46KB) and `test/maxcut/lanal.md` (358
   lines) are both *not documentation*** — both are accidental dumps of
   raw Claude Code terminal-session transcripts (each starts with the
   Claude Code ASCII banner; `lanal.md` ends mid-transcript with a
   `/rename loopanalysis` command, which is very likely how
   `loopanalysis.md` came to exist as a second, separate copy/session).
   Neither was edited or removed as part of this pass — flagging their
   real nature is the fix that matters; deleting either is a judgment
   call for the project owner, not made here.
3. **`readme.txt`** (root) is one line, a link to AMD's external LLVM
   loop-terminology docs — not a project description, left as-is.
4. **`clique_impl` staying classical-only** is the one bridge this
   session's kernel-first rollout deliberately didn't touch — not from
   lack of time, but because clique's kernel answers are only provably
   *valid*, never provably *maximum* (unlike TSP, there's no small-N
   degenerate case where validity implies optimality), so the same
   mechanical pattern applied to maxcut/tsp/factor would either weaken
   `clique_impl`'s correctness contract or require real additional design
   work. See the bridge-layer section above for the full reasoning.

## Appendix: directory map

```
loopanalysis/
  pass.cpp                    registration hub (legacy MinPass matcher removed)
  maxcut_pass.cpp  tsp_pass.cpp  kcolor_pass.cpp  clique_pass.cpp  factor_pass.cpp
  CMakeLists.txt               builds MinPass.so; check-<name> lit targets
  c2cudaq -> /home/def3r/def3r/SIGSEGV/QuantumComp/c2cudaq   (symlink)
  test/
    maxcut/ tsp/ kcolor/ clique/ factor/     lit suites (README.md each)
    integration/                              e2e compile+link+run (README.md)
    <~30 loose scratch files, undocumented>   see "test/ -- what's important" above
    viz/                                      empty
  analysis/
    maxcut/    maxcut.md, report.html, stress source          (stress-tested)
    tsp/       tsp.md, report.html, stress source              (stress-tested)
    kcolor/    report.html, raw/staged .ll files, no .md       (stress-tested)
    clique/    clique.md, report.html, .ll files, stress source (stress-tested)
    factor/    factor.md, report (published Artifact), stress sources (stress-tested)
  tools/
    qoffload-clang++, README.md               single-command wrapper
  PROJECT_INDEX.md                            this file
  loopanalysis.md                             NOT docs -- stray transcript dump
  readme.txt                                  NOT docs -- one external link

c2cudaq/  (separate repo, symlinked in)
  src/bridge.cpp        the five *_impl bridge functions
  src/grover.cpp         c2q_factor (Grover)
  src/qaoa.cpp            c2q_maxcut / c2q_kcolor / c2q_clique / c2q_tsp (QAOA)
  README.md               kernel API reference
  claude.md               unrelated in-progress SQIF design doc
  tests/                  kernel-level correctness tests (not this project's)
```
