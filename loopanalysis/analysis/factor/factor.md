# Factor — status notes

Explored whether `c2q_factor` (c2cudaq's Grover-based factoring kernel, in
`QuantumComp/c2cudaq/src/grover.cpp`) is worth a matching `factor_pass.cpp`.
Stress-tested the kernel first (§2 below) — on current simulated hardware
the answer was "not worth it," classical trial division wins by 5-10
orders of magnitude across the entire `n ≤ 127` reachable range. Built
anyway: the ceiling is a simulator artifact, not an algorithmic one, and
Grover's real quadratic advantage only shows up on hardware this project
isn't targeting yet. `factor_pass.cpp` is implemented and verified (§1)
so the replacement is ready the day that stops being true.

## 1. What the kernel actually does

`c2q_factor(n)` runs a Grover search over superposed pairs `(a, b)`: a
QFT-based multiplier circuit computes `acc = a * b` in superposition, an
oracle phase-flips states where `acc == n`, and a diffuser amplifies those
states — the standard Grover shape, iterated `~(π/4)·√(N/M)` times. This
was confirmed by reading the multiply block's own code and comments
(`acc = work_a * work_b`) after an initial guess (single-variable trial
division via modulus) turned out to be wrong — the oracle evaluates
products of *pairs*, not `n % d == 0` against one candidate at a time.

Constraints, read directly from `c2q_factor`'s body:
- Throws if `n < 4`.
- `num_result` = bit-length of `n`; `num_state = num_result - 1`;
  `total_q = 4*num_state + 1`.
- Throws if `total_q > 28` → hard practical ceiling of **`n ≤ 127`**.
- No seed parameter — same input always drives the same search.

## 1a. The classical target — brute-force pair search, not trial division

Since the oracle evaluates `a * b == n` over superposed **pairs**, the
classical code a matcher should target is the classical version of that
same unstructured search — a nested double loop over candidate pairs —
not trial division (`for d in 2..√n: if n % d == 0`), which is a
*smarter* classical algorithm than what Grover's quadratic speedup
applies to. Target, `test/factor.cpp`:

```cpp
bool bruteForceFactor(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n) {
        outA = a;
        outB = b;
        return true;
      }
    }
  }
  outA = 1;
  outB = n;
  return false;
}
```

Four scope questions, resolved before writing the matcher:

1. **Loop bounds** — both loops must bound against the *same* `Argument`
   as the product-equality check (three uses of the identical value).
   This is the real signature of "searches the full n×n space"; a
   triangular-bound variant (`a*a <= n`) is a structurally different
   shape and isn't matched — same "known limitation, not solved"
   precedent as clique's `std::max`-only accumulator decision.
2. **Early return vs. exhaustive scan** — matches the natural
   `return true` shape a boolean "does a pair exist" function produces;
   an exhaustive-scan-then-argmin variant belongs to a different problem
   (find *all* pairs) and isn't targeted.
3. **Fallback shape — kept tight.** The "not found" merge edge must carry
   exactly `(ConstantInt 1, NArg)`, not any constant pair. Matches
   `decode_factors`' expected prime-fallback convention; a hand-written
   `(0, 0)` sentinel wouldn't match, by design.
4. **Nested loop only** — a flat single loop decomposing one index via
   `i / bound, i % bound` is a different, unnatural IR shape and isn't
   supported; nobody writes factor search that way from scratch.

## 1b. The actual shape — architecture and one genuinely new wrinkle

`factor_pass.cpp` is a **`FunctionPass`**, like `tsp_pass.cpp`/
`maxcut_pass.cpp` — not a Module pass like kcolor/clique. There's no
self-recursion or inter-procedural call-site hunting needed here; the
whole search lives in one function, so matching and replacement both
happen in place via `LoopInfo`.

Confirmed by reading the canonicalized IR directly:

1. **Fully SSA — no memory-resident accumulator at all.** Unlike every
   other pass here, `outA`/`outB` are output *parameters*, not locals
   `std::max`/`std::min` forces into an alloca. Both loop counters stay
   phis straight through `mem2reg`; the matcher never has to trace a
   load-store shape, only phi backedges and `icmp` operands.
2. **`Loop::getLoopPreheader()` fails here — use `getLoopPredecessor()`
   instead.** The inner loop's "preheader" is the outer loop's *header*
   itself (control falls straight from `a < n` into the inner loop with
   no init block between them) — but `getLoopPreheader()` additionally
   requires that single external predecessor to have exactly **one**
   successor, and the outer header has two (into the inner loop, and out
   to the exit block, via its own bound check). `getLoopPredecessor()`
   only requires a unique external predecessor, with no successor-count
   restriction, and correctly returns the outer header. This is a wrinkle
   none of the other four passes hit, because none of them have an inner
   loop whose entire "preheader" role is played by a block that's
   simultaneously exiting an outer loop.
3. **The exit block owns all three of the function's outputs at once —
   both stores AND the return value** — not just one accumulator's
   post-loop value the way `tsp_impl`'s replacement is a single RAUW.
   `performReplacement` has to explicitly erase the two merge phis, the
   two stores, and rebuild the `ret` around the call's own result, rather
   than RAUW-ing external users of one alloca.

## 1c. Running it — the exact command, and pass order

Same reduced pipeline as `clique_pass.cpp`, confirmed the same way:
compiled with and without `loop-simplify,lcssa,indvars` and diffed the
IR. `indvars` inserts `%smax = call i32 @llvm.smax.i32(i32 %n, i32 2)`
and rewrites the outer loop's exit test to `icmp ne %a.0, %smax`, burying
`%n` inside the `smax` call instead of leaving it as a direct `icmp`
operand — the same failure mode documented in `clique.md` §1c, hitting
the same `NArg` check here.

```bash
clang -S -emit-llvm -O0 -fno-inline -Xclang -disable-O0-optnone \
  -fno-discard-value-names test/factor.cpp -o factor.ll

opt -passes="sroa,mem2reg,simplifycfg,instcombine<no-verify-fixpoint>,simplifycfg,instcombine" \
  factor.ll -S -o factor_opt.ll

opt -load-pass-plugin=build/MinPass.so -passes="factor-pass" \
  -debug-only=factor-cpp -S factor_opt.ll -o factor_replaced.ll
```

`-debug-only=factor-cpp`, not any other pass's debug type (`DEBUG_TYPE`
is `"factor-cpp"`). Verified: matches and replaces `bruteForceFactor` in
`test/factor.cpp` with a call to `@factor_impl`, produces verifier-clean
IR, and — cross-checked against `clique/basic.cpp`, `kcolor/basic.cpp`,
`tsp/basic.cpp`, `maxcut/basic.cpp` — fires on none of them.

## 2. Kernel stress test

Full report with charts: **[artifact link — factoring kernel stress
test](https://claude.ai/code/artifact/dd88e2d3-df47-4980-a909-139f929bfb48)**

Source: `analysis/factor/factor_fast.cpp` (tiered sampling across all 5
qubit tiers + a 5x repeat of `n=15` for a determinism check),
`analysis/factor/factor_tier5.cpp` (isolated single-shot spot checks at
the expensive 25-qubit tier), `analysis/factor/factor_boundary.cpp`
(confirms the `n=128` throw). Raw output: `analysis/factor/stress_output.txt`.

An earlier, exhaustive version of this test (every composite 4..127 at 3
trials each) was abandoned — see the "design note" in
`stress_output.txt` for why: qubit count is a step function of `n`'s
bit-length, so an exhaustive sweep queues ~150 expensive 25-qubit calls
before reaching any cheap ones, with no output flushing to show progress.
It looked like a hang; it wasn't — `ps aux` confirmed 99.5% CPU the whole
time. Killed after 11+ minutes, replaced with per-tier sampling and
explicit `endl` flushing.

### Findings

- **Correct, every trial — 26/26.** Composites and primes both handled
  correctly across all five qubit tiers (9/13/17/21/25 qubits), plus
  `n=15` repeated 5x with an identical answer every time. Unlike kcolor
  and clique (QAOA — sound but *incomplete*, misses valid answers as
  problems grow), this kernel stayed both sound *and* complete throughout
  its entire testable range. Correctness never degrades with size, only
  wall-clock cost does.
- **Cost climbs in qubit-count steps, and the steps are steep.**
  `total_q = 4*num_state + 1` jumps a whole tier every time `n` crosses a
  power of two. Measured averages: ~1.5ms (9q, n=4-7) → ~3.0ms (13q,
  n=8-15) → ~13.6ms (17q, n=16-31) → ~1.03s (21q, n=32-63) → ~46.0s (25q,
  n=64-127, spot-checked at n=85 and n=91). Roughly 30,000x between
  cheapest and most expensive tested tier.
- **The 28-qubit simulator ceiling sits exactly one tier past the last
  tested one.** `c2q_factor(128)` throws immediately, before any
  simulation runs: `"c2q_factor: n requires 29 qubits (simulator limit
  28); safe range is n <= 127 (num_result <= 7)"`. Confirmed directly, not
  inferred from the formula.
- **Classical trial division is not a fair fight.** Measured on the same
  machine: ~6ns average per `factor(n)` call across the whole n=4-127
  range via `O(√n)` trial division (at most 11 candidate divisors at
  n=127) — fast enough to be competing with clock resolution, not with
  any real computational limit. That makes the quantum kernel ~250,000x
  slower at its cheapest tier and **~7.6 billion times slower** at its
  most expensive (n=64-127).
- **This is the opposite shape from maxcut/TSP/kcolor/clique.** Those
  four target genuinely NP-hard problems where brute force explodes well
  inside sizes someone would actually want to run (`19!` orderings for a
  20-city TSP tour, `2^25` subsets for a 25-vertex clique check).
  Factoring `n ≤ 127` has no such wall — the "hard case" is eleven
  modulus operations. The kernel isn't competing against an expensive
  classical algorithm at any size it can actually reach.

### Practical limits

| Regime | n range | Correctness | Speed |
|---|---|---|---|
| Fast | 4–15 | 100% | 1–3ms |
| Fine | 16–63 | 100% | 14ms–1.0s |
| Slow but correct | 64–127 | 100% | ~46s |
| Not reachable | ≥ 128 | — | throws — needs 29+ qubits, simulator caps at 28 |

### Verdict: not worth it on today's hardware — wired in anyway, on purpose

Correctness was never the constraint — this is the most reliable kernel
in the set so far, sound *and* complete everywhere it was tested. On a
state-vector simulator, routing a structurally valid Grover target to
`c2q_factor` costs a quarter-million to 7.6-billion times more wall-clock
than the brute-force loop it replaces, with no size regime inside
`n ≤ 127` where that trade pays off on pure performance grounds. But
that ceiling is the simulator's, not the algorithm's — Grover's real
quadratic edge shows up at bit-lengths this project was never going to
reach in simulation anyway.

**Updated project decision**: `factor_impl` (`c2cudaq/src/bridge.cpp`)
now calls `c2q_factor` first whenever `4 ≤ n ≤ 127` — the kernel's
entire verified-correct range — rather than staying classical-only.
Performance was deliberately set aside as the deciding factor: the point
is to actually exercise the quantum path wherever it's proven correct,
not to wait for it to also be fast. Classical brute force remains the
fallback for `n` outside that range and for the (empirically
unobserved, but not impossible) case of an ambiguous kernel "no factor
found" answer — same mandatory-fallback-on-ambiguity convention every
other bridge in this project uses. A positive kernel answer is
self-verified for free (`a*b==n`) before being trusted, so this doesn't
weaken the classical-only version's correctness guarantee anywhere in
range; it only changes which path gets tried first, and how long that
path takes.

## Next steps (when revisited)

1. ~~Write `factor_pass.cpp`.~~ Done — see §1a for the target/scope
   decisions, §1b for the two genuinely new implementation wrinkles vs.
   the other four passes, §1c for the exact command/pipeline.
2. ~~Build the `test/factor/` lit test suite.~~ Done — 20 tests (18
   REJECT, 1 DETECT-with-sibling, 1 XFAIL), `cmake --build build --target
   check-factor` green. See `test/factor/README.md` for the full
   breakdown, including two things that turned out differently than
   planned once actually compiled: a true "two different bound arguments"
   false-positive probe isn't constructible at all (the fixed 3-argument
   signature rules it out by construction), and instcombine's
   canonicalization erased more "distinct" surface shapes (inverted
   predicates, `<=`) than expected — both documented in the relevant test
   files rather than papered over.
3. ~~Build `factor_impl` in `c2cudaq/src/bridge.cpp`.~~ Done — originally
   classical-only, matching `bruteForceFactor`'s own semantics exactly.
   Verified end-to-end via `test/integration/factor_e2e.cpp` /
   `test/integration/run_factor_integration.sh`: `factor-pass` fires,
   links against `libc2cudaq.a`, runs, and correctly factors 91 = 7×13.
   No regressions in the other four integration tests after rebuilding
   both `libc2cudaq.a` and `MinPass.so`.
4. ~~Wire `c2q_factor` into `factor_impl`.~~ Done — see the updated
   verdict above. Kernel-first for `4 ≤ n ≤ 127`, self-verified
   (`a*b==n`), classical fallback outside that range or on an ambiguous
   kernel "no factor" answer. Re-verified end-to-end after the change:
   `factor_e2e` (n=91, now kernel-routed) takes 46.9s, matching this
   doc's own §2 GPU timing table exactly — that number was only
   reproducible once a separate, unrelated bug was also found and fixed
   (`test/integration/run_factor_integration.sh` and every sibling script
   were hardcoding the CPU backend regardless of GPU presence; see
   `analysis/tsp/tsp.md`'s next-steps item 5 for the full story, since it
   was found while re-testing `tsp_impl`'s equivalent change).
