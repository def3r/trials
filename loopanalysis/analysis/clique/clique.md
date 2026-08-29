# Clique — status notes

Picks up after the kcolor pass/kernel work. Three things done so far:
`test/clique.cpp` refactored to match the `tsp.cpp`/`kcolor.cpp` convention,
`c2q_clique` (c2cudaq's QAOA clique kernel) stress-tested to know its limits,
and `clique_pass.cpp` written and verified working against real IR. No
`test/clique/` lit suite yet.

## 1. `test/clique.cpp` refactor

Was raw global arrays (`int graph[100][100]`, `int store[100]`, `int n`,
plus an unused `d[100]` degree array) — converted to
`vector<vector<int>>& graph`, `vector<int>& clique`, `int N` passed as
parameters, matching how `kcolor.cpp`/`tsp.cpp` take their input. This is
the main precondition for IR pattern-matching to even be tractable — global
arrays show up as bare `@graph`/`@store` references with no argument-based
container to trace; `vector<vector<int>>&` shows up as the same
`operator[]`-call chain shape `maxcut_pass.cpp`/`tsp_pass.cpp`/
`kcolor_pass.cpp` already know how to recognise.

New shape:

```cpp
bool isClique(int size, vector<int>& clique, vector<vector<int>>& graph);
int  maxCliques(int start, vector<int>& clique, int size, int N, vector<vector<int>>& graph);
int  findMaxClique(vector<vector<int>>& graph, int N);
```

Verified correct on two cases: the reference K4 graph (max clique 4) and a
5-vertex triangle+pendant+isolated-vertex graph (max clique 3, not the
trivial complete-graph answer). Confirmed via raw IR dump that no global
state leaked through.

## 1a. Which passes identify it — verified against actual IR

Same empirical approach as the kcolor CFG investigation: compiled
`test/clique.cpp` through `sroa,mem2reg,simplifycfg,instcombine` (`stageA`),
diffed against `+loop-simplify,lcssa,indvars` (`stageB`), and against
`+early-cse` (`stageC`). Working files: `analysis/clique/clique_ex.ll`,
`clique_stageA.ll`, `clique_stageB.ll`, `clique_stageC.ll`.

**Passes that help, same verdict as kcolor:**
- `sroa` + `mem2reg` — essential.
- `simplifycfg` + `instcombine` — essential, collapses the shape.
- `LoopInfo` — needed for `maxCliques`'s own `for (v = start; v < N; v++)`
  loop (to get `v`'s phi). **Not** needed for `isClique`'s internal nested
  loop — same as kcolor's `isSafe`, `isClique` is matched as an opaque
  guard call, its own internals never inspected.
- `loop-simplify` — confirmed unnecessary (loop already has a single
  preheader/latch post-simplifycfg, verified via empty diff on that front).
- `lcssa` — confirmed unnecessary (`grep -c lcssa` on the fully-piped IR:
  `0`). `best` is memory-resident, not an SSA value crossing the loop
  boundary.
- `indvars` — confirmed unhelpful, same failure mode as kcolor: widens
  types, inserts an `llvm.smax` clamp, and for `isClique`'s nested loop
  specifically adds a whole extra block splitting `for.body`/`for.inc`
  that isn't there without it. Doesn't matter for matching since
  `isClique`'s internals are opaque anyway, but confirms the same "don't
  run indvars" conclusion for the parts that DO matter.
- `ScalarEvolution`/`InductionDescriptor` — considered, not needed. The
  loop's induction variable is the same canonical
  `phi [start, preheader], [inc, latch]` + `icmp slt` + `add nsw ..., 1`
  shape `tsp_pass.cpp`/`maxcut_pass.cpp` already match by hand; using SE
  would be a real analysis but an inconsistent one relative to the rest of
  this codebase, which never reaches for it (`pass.cpp`'s original MinPass
  is the only place that does, and it isn't the pattern the other three
  passes settled on).
- **`early-cse` — new, and genuinely useful here, unlike for the other
  three passes.** `size + 1` is computed **three separate times**
  (`isClique`'s call, the `std::max(best, size+1)` temporary, and the
  recursive call's `size` argument) as three distinct, un-CSE'd SSA values
  in `stageA`. `early-cse` collapses all three into one shared value
  (verified: `stageC` diff shows exactly that, `opt -passes=verify`
  clean). Without it, a matcher has to re-derive "is this the same
  `size+1` expression" independently at three call sites via structural
  comparison; with it, one shared SSA value can be checked once and reused
  by identity everywhere else.

## 1b. The actual shape — and what's genuinely new vs. kcolor_pass.cpp

Confirmed by reading `maxCliques`'s and `findMaxClique`'s canonicalized IR
directly (not just reasoning about the source):

1. **`best` stays memory-resident for the same reason `kcolor`'s
   `currCost`/`minCost` did**: `std::max<int>(const int&, const int&)`
   takes its arguments by reference, so SROA/mem2reg can't promote it.
   Confirmed in the IR — `%best = alloca i32`, load-add-store shape
   throughout, exactly like kcolor's accumulators.

2. **The self-call mixes two different "+1" sources in the same call**,
   not just one: the `start` argument slot is `%v.0 + 1` (**the enclosing
   loop's own phi**, not a formal parameter), while the `size` argument
   slot is `%size + 1` (a formal parameter — same style as kcolor's
   `node + 1`). `matchSolve()`'s existing check (self-call argument must
   be `FormalArg + 1`) only covers the second case. A clique matcher needs
   to check each self-call argument against **two** possible "+1" sources:
   known formal parameters, and the candidate loop's own induction
   variable phi.

3. **The guard call's argument is itself a derived value**
   (`isClique(size + 1, clique, graph)`), not a raw parameter the way
   kcolor's `isSafe(node, ...)` took `node` directly. With `early-cse` in
   the pipeline this is the same shared SSA value used everywhere else, so
   it's one direct identity check, not a re-derivation.

4. **No backtrack store at all.** kcolor's `color[node] = 0` on the
   failure edge has no counterpart here — `clique[size]` just gets
   overwritten by the next loop iteration's assign, so there's nothing to
   undo. A clique matcher must NOT require a backtrack pair; it's simply
   absent from this shape, not a variant that failed to include one.

5. **The accumulator updates twice per candidate and the loop never exits
   early** — `best = max(best, size+1)` (accept this extension) **and**
   `best = max(best, maxCliques(...))` (keep searching deeper), both
   updating the same memory-resident `best`, and the loop always continues
   to the next candidate regardless of what the recursive call returned.
   This is structurally closer to `maxcut_pass.cpp`'s `MaxCompare`/
   `MaxUpdatePhi`/`std::min`-or-`std::max`-call recognition (running-best
   across all iterations, no early exit) than to kcolor's early-return
   boolean decision. The two `std::max` call sites should reuse that
   detection logic rather than kcolor's single-early-return pattern.

6. **Phase 2 (top-level call detection) is easier, not harder.**
   `findMaxClique` calls `maxCliques(0, clique, 0, N, graph)` with **two**
   literal-zero arguments (`start` and `size`), not just one — a stronger
   anchor than kcolor's single `node == 0` signal. Confirmed via
   `findMaxClique`'s IR: `call ... i32 noundef 0, ptr ..., i32 noundef 0,
   ...`. Also confirmed it's an `invoke` (same mechanical wrinkle as
   kcolor's `graphColoring`, same fix already exists in
   `performReplacement`).

Everything else about `kcolor_pass.cpp`'s approach — `operator[]`
recognition via `stripToContainerSource`, the side-effect gates on the
guard function and the recursive function, the Module-pass structure for
inter-procedural matching, the invoke→call replacement mechanics — carries
over directly.

## 1c. Running it — the exact command, and pass order

`clique_pass.cpp` is implemented, registered as `clique-pass`, and verified
matching/replacing correctly against real IR (including a cross-contamination
check: it does **not** fire on kcolor's `basic_opt.ll`).

**The command that works, start to finish — no `llvm-extract` needed.**
Unlike the `test/kcolor/`/`test/tsp/` suites (which extract just the target
functions to keep test files small), `clique-pass` is a Module pass and
scans every function regardless, so it's fine to run it straight on the
whole translation unit:

```bash
clang -S -emit-llvm -O0 -fno-inline -Xclang -disable-O0-optnone \
  -fno-discard-value-names test/clique.cpp -o clique.ll

opt -passes="sroa,mem2reg,simplifycfg,instcombine<no-verify-fixpoint>,simplifycfg,instcombine" \
  clique.ll -S -o clique_opt.ll

opt -load-pass-plugin=build/MinPass.so -passes="clique-pass" \
  -debug-only=clique-cpp -disable-output clique_opt.ll
```

Three things about this that aren't obvious and cost real debugging time
the first time through:

- **`instcombine<no-verify-fixpoint>` is required when compiling the whole
  file** (not just the three target functions) — without it, instcombine
  hits a hard error on one of the `std::vector` template instantiations
  (`_M_realloc_insert` et al.) elsewhere in the translation unit:
  `Instruction Combining ... did not reach a fixpoint`. Not needed if you
  `llvm-extract` down to just `isClique`/`maxCliques`/`findMaxClique`
  first (as `clique_ex.ll` in this directory does) — the fixpoint issue
  lives entirely in code the pass never looks at.
- **`-debug-only=clique-cpp`, not some other pass's debug type.** This
  pass's `#define DEBUG_TYPE` is `"clique-cpp"`. Using e.g.
  `-debug-only=maxcut-cpp` by copy-paste habit gives total silence even
  when the pass *does* attempt a match and reject at a gate — it looks
  identical to "didn't run at all," which is exactly the confusing report
  that prompted this section.
- **`early-cse` is genuinely optional and position-independent** — tested
  it in four different places in the pipeline (omitted; last; between the
  first instcombine and the final simplifycfg; right after mem2reg) and
  all four match and replace identically. This isn't a coincidence: per
  the "structural re-derivation everywhere" decision (§1d, B below), the
  matcher never depends on `early-cse` having merged anything, so where
  you put it (or whether you include it at all) can't change the outcome.
  Include it if you want cleaner IR to read while debugging; the pass
  doesn't care either way.

**What you must NOT add: `loop-simplify`, `lcssa`, `indvars`.** Confirmed
by reproducing a real failure report: running
`sroa,mem2reg,loop-simplify,lcssa,indvars,simplifycfg,instcombine<no-verify-fixpoint>,simplifycfg,instcombine`
makes the match fail. Root cause is `indvars` specifically — it clamps the
loop's trip count through a new call, `%smax = call i32
@llvm.smax.i32(i32 %N, i32 %start)`, and rewrites the loop's exit check
from `icmp slt %v.0, %N` to `icmp eq %v.0, %smax`. `NArg` detection
(§1b) only recognises a **raw `Argument`** as a direct operand of that
comparison; `%N` is still in there, but buried inside the `smax` call
instead of sitting directly in the `icmp`, so the check finds nothing and
the whole match fails at that step. Exactly the same failure mode already
documented for `kcolor_pass.cpp` in §1a above — this is why those three
passes are excluded from the recommended pipeline, not an oversight.

## 1d. Three decisions made while implementing it

Resolved before writing any code, recorded here so the reasoning doesn't
have to be reconstructed later:

**A. Phase 2 top-level call detection requires BOTH `start == 0` AND
`size == 0`, not just one.** `findMaxClique` calls `maxCliques(0, clique,
0, N, graph)` — both are literal zero. Requiring both is a stronger,
more specific anchor than either alone; the tradeoff accepted is that a
hypothetical "resume search from a partially-built clique" call site
(`start == 0`, `size != 0`) wouldn't be picked up as a replacement target
— not a pattern anything currently produces.

**B. No pipeline dependency on `early-cse` — every "is this the same
`size + 1` expression" check is structurally re-derived independently at
each use site** (the guard call's argument, the accumulator's
accept-update operand, the self-call's `size` slot), rather than relying
on `early-cse` having already merged them into one shared SSA value. More
matcher code (`isAddOne()` gets called three separate times instead of
one identity comparison), but consistent with how `maxcut_pass.cpp`/
`tsp_pass.cpp`/`kcolor_pass.cpp` already work, and — as borne out in §1c
— means the pass's correctness doesn't depend on a specific pipeline
stage being present. This is *why* the "what pass order is required"
question in §1c has the answer "`early-cse` position doesn't matter, and
you can skip it entirely": it was a deliberate design choice, not an
accident.

**C. Only the `std::max` call form is recognised for the running-max
accumulator, not a hand-written comparison.** `best` only stays
memory-resident because `std::max<int>(const int&, const int&)` takes its
arguments by reference — same mechanism as kcolor's `std::min`. TSP's
`min_cmp_form.cpp` already found that a hand-written `if (x > best) best =
x;` bypasses this entirely: with nothing forcing the address to escape,
`mem2reg` promotes the accumulator straight to an SSA phi, and the
load-store shape the matcher looks for is never there. A hypothetical
`clique.cpp` variant using a manual comparison instead of `std::max` would
hit that identical gap. Not solved here — documented as a known
limitation, same as `min_cmp_form.cpp` does for TSP, rather than
speculatively building support for a form not confirmed reachable via
natural compilation.

## 2. `c2q_clique` kernel stress test

Full report with charts: **[artifact link — clique kernel stress
test](https://claude.ai/code/artifact/3c524cba-18dd-45b0-a117-dd37cac1f5c1)**

Source: `analysis/clique/clique_stress.cpp`. Raw output:
`analysis/clique/stress_output.txt`.

### Findings

- **Sound, every trial.** 0 false positives across 27 infeasible-case runs
  (star graphs and empty graphs, requesting cliques the graph provably
  doesn't have). Same property confirmed for kcolor earlier —
  `decode_clique` verifies every returned set before reporting a size.
- **Completeness depends on problem *shape*, not just qubit count.** A
  trivial case (complete graph, any subset is automatically valid) held
  70–90% hit rate up to N=16 at `layers=2`. A realistic case (a genuine
  size-4 clique planted in an otherwise sparse N-vertex graph) collapsed to
  ~0% by N=8, same qubit range. Qubit count alone doesn't predict
  difficulty; the QUBO landscape's shape does.
- **The wall-clock cliff arrives well before the 28-qubit hard limit
  `check_qubit_limit` enforces.** N=16→18 roughly quadruples runtime; N=20
  took 52s and N=22 took 4 minutes, for a *single* seed — and both of
  those also failed to find the target clique. Nobody gets near 28 qubits
  in practice.
- **More QAOA depth is a cheap fix at fixed N.** `layers=2→4` did nothing
  at N=16 (40%→40%); `layers=6` hit 100%, for <2× the time — depth cost is
  linear (doesn't touch qubit count), unlike qubit count itself. **Not yet
  tested**: whether extra depth also rescues the hard (planted-clique)
  case the way it rescued the trivial one — natural next experiment if
  this becomes a real dependency.
- **`c2q_clique(g, k=-1)` doesn't always return exactly `K`.** Default
  targets `K = N-1`; on dense graphs the optimizer can drift to selecting
  *more* than K vertices, since the QUBO only softly penalizes deviating
  from the target count (rather than hard-constraining it), and extra
  vertices in a dense graph often stay clique-valid. Not a soundness
  issue — whatever's returned is still verified — but a caller expecting
  `objective == K` exactly would be surprised on dense inputs.

### Practical limits

| Regime | N (qubits) | Verdict |
|---|---|---|
| Reliable | ≤ 6–8 | high hit rate, sub-second |
| Usable, needs care | 10–16 | 70–90% at layers=2 on easy cases; near-zero on hard ones — raise layers |
| Impractical | 18–22 | tens of seconds to minutes per run, unreliable even then |
| Untested | > 22 | extrapolated cost: hours+ per run |

### For the eventual `clique_impl` bridge

Same kernel-first / mandatory-classical-fallback shape as the kcolor
bridge (`c2cudaq/src/bridge.cpp`'s `kcolor_impl`): trust an immediate
success, always re-verify classically on failure/timeout, never let the
kernel be the sole source of a "no" answer. But the size cutoff should
land well under 16 qubits, not near kcolor's 24 — the hard-case failure
here sets in far earlier than the qubit ceiling would suggest, and the
wall-clock cost alone makes a generous cutoff a bad trade even before
reliability is considered.

## Next steps (when revisited)

1. ~~Write `clique_pass.cpp`.~~ Done — see §1c for the exact
   command/pipeline and §1d for the three implementation decisions.
2. Build the `test/clique/` lit test suite mirroring `test/kcolor/`'s
   structure (`update.py`, `README.md`, `lit.cfg.py`, CHECK-directive-driven
   `.cpp` files per matcher step) — same category breakdown kcolor's suite
   used (basic DETECT, per-gate REJECTs, arity/return-type scope-boundary
   tests, a Phase-2 multi-call-site test), adapted for the two new
   decisions in §1d and the loop-phi/formal-param self-call classification
   in §1b.2.
3. Test whether more QAOA layers helps the planted-clique (hard) case,
   not just the trivial complete-graph case.
4. Decide the concrete qubit/size cutoff for `clique_impl` given the
   findings above (candidate: somewhere around N=10–12, well short of
   kcolor's 24).
5. Build `clique_impl` in `c2cudaq/src/bridge.cpp` + a classical
   `findMaxClique`-based fallback, mirroring `kcolor_impl`'s structure.
