# MaxCut — status notes

Stress-tested `c2q_maxcut` (c2cudaq's QAOA MaxCut kernel) to fill the gap
flagged in `PROJECT_INDEX.md`'s "Open, deliberately unaddressed items" —
this was the one kernel of the five with no analysis directory and no
stated status at all (not even "deferred," unlike TSP). `maxcut_pass.cpp`
and its bridge (`maxcut_impl`, kernel-first, `c2cudaq/src/bridge.cpp`)
were both already built and working before this; this fills in whether
that kernel-first choice was actually justified.

## 1. Kernel stress test

Full report with charts: **[artifact link — MaxCut kernel stress
test](https://claude.ai/code/artifact/40ca8f53-1ff9-434f-aff6-0fe68ceaa419)**

Source: `analysis/maxcut/maxcut_stress.cpp`. Raw output:
`analysis/maxcut/stress_output.txt`.

### Structural note: there is no invalid-answer mode

Confirmed directly (0/10 negative objectives on a random N=10 graph) and
true by construction: `decode_maxcut` just sums cut-edge weight for
whatever bit assignment comes back. Every bitstring is a valid 2-partition
— there's no constraint to violate, so no `-1 = invalid` case exists the
way it does for clique/kcolor/TSP. This makes MaxCut the only one of the
four QAOA-based kernels with no soundness axis to evaluate at all; the
entire stress test is about *completeness* (does it find the optimum) and
cost.

### Findings

- **Graph regularity predicts completeness far better than qubit count
  does.** Three graphs at matched N, `layers=2`, 10 seeds/point, ground
  truth via exhaustive classical search (or closed-form for the
  structured cases):
  - **Cycle graphs**: 100% hit rate at every N tested, 4 through 16.
  - **Complete graphs**: 80–100% through N=14, dropping to 60% at N=16.
  - **Random sparse graphs** (density 0.4, the realistic case): 100% at
    N=4, falls to 30–60% by N=6–12, **0% by N=14**.
  Same qubit count, three completely different outcomes — the sharpest
  version of the "shape, not size" finding also seen in clique's stress
  test, sharper here specifically because there's no soundness
  confound to muddy the comparison.
- **Wall-clock cost grows exponentially, same shape as every other
  kernel tested — but roughly 3× cheaper at matched qubit count.**
  N=22 took 82s here vs. clique's 240s for the same N. Likely explanation:
  MaxCut's QUBO has no penalty terms (nothing to constrain, since every
  bitstring is valid), so the cost Hamiltonian is simpler than clique's
  size/validity-penalized one — and that shows up as real wall-clock
  savings, not just a correctness difference.
- **Unlike clique, more QAOA depth does NOT reliably rescue the hard
  case.** Clique's stress test found `layers=2→6` turned a 40% hit rate
  into 100% at fixed N. Repeating that lever here (random graph, N=14,
  the point completeness had just collapsed) gave 20% → 40% → 20% —
  noise around a low baseline, not a trend. The two kernels' QUBO
  landscapes respond differently to circuit depth; whatever makes this
  random-graph instance hard isn't primarily a depth problem.
- **Single-seed spot checks at N=18–22 all missed the true optimum** —
  by 1 edge at N=22 (117 vs. 121) up to 25 edges at N=18 (56 vs. 81) —
  so, same as clique/kcolor, extra wall-clock time past the tested
  10-seed ceiling doesn't reliably buy a better answer either.

### Practical limits

| Regime | N (qubits) | Verdict |
|---|---|---|
| Reliable, structured graphs | ≤ 16 (tested ceiling) | cycle graphs: 100% throughout |
| Usable, needs care | ≤ 10–12 | complete graphs: 80–100%; random graphs: 30–60% |
| Unreliable | 14–16 | random-graph hit rate reaches 0%; extra depth doesn't fix it |
| Impractical | 18–22 | 10–82s per run, single spot checks all missed the optimum |

### For the existing `maxcut_impl` bridge

**Updated.** `maxcut_impl` in `c2cudaq/src/bridge.cpp` used to be
kernel-first with no classical fallback at all — the caution this stress
test originally raised (no soundness backstop the way
`decode_kcolor`/`decode_clique` provide, so a kernel-first call on an
irregular graph past ~N=12–14 could silently return a *valid but
suboptimal* cut with no signal that it wasn't the true maximum).

Fixed: below `kMaxCutExactCutoff = 16` nodes, the bridge now runs an
exhaustive classical enumeration (2^N partitions, cheap at this size)
*alongside* the kernel call and keeps whichever cut is better — a free
correctness floor, since classical is exact there and can only be
matched or beaten, never silently lost to. Above N=16, the bridge stays
kernel-only, as before; the missing-invalid-signal risk documented above
is still real past that gate, left as a deliberate, documented tradeoff
rather than an oversight (raising the cutoff further trades wall-clock
for a wider safety net, and is a size choice, not a correctness fix).

## Next steps (when revisited)

1. ~~Stress-test `c2q_maxcut`.~~ Done — see above.
2. ~~Add a fallback/safety-net to `maxcut_impl`.~~ Done — see above.
3. Test whether a denser random graph (this test used density 0.4)
   degrades faster or slower than the sparse case tested here — not
   covered, and density is a plausible second axis of "shape" beyond
   regular-vs-irregular.
