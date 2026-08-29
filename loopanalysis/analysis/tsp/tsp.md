# TSP — status notes

Stress-tested `c2q_tsp` (c2cudaq's QAOA TSP kernel) to fill the gap
flagged in `PROJECT_INDEX.md`. Unlike MaxCut, this one was already
explicitly marked "deferred" — `tsp_impl` in `c2cudaq/src/bridge.cpp`
stays classical-only, with a comment saying `c2q_tsp` "hasn't been
stress-tested for reliability or wall-clock limits." This fills that in.
**Confirms the classical-only decision was correct** — if anything, more
so than expected.

## 1. Kernel stress test

Full report with charts: **[artifact link — TSP kernel stress
test](https://claude.ai/code/artifact/8fc1e6f2-a64b-4cdd-abcf-2e594e237c94)**

Source: `analysis/tsp/tsp_stress.cpp`. Raw output:
`analysis/tsp/stress_output.txt`.

**This stress test was cut short on hardware-safety grounds and that
matters for how to read the data below.** The N=5 (25-qubit) point ran a
single seed for over 20 minutes at 100% GPU utilization and 78°C with no
sign of completing, and was killed rather than pushed further. N=3 and
N=4 both completed cleanly and are reported in full; N=5 is reported as
"attempted, not completed" rather than extrapolated or estimated.

### Structural note: qubit scaling is the headline finding, not a footnote

TSP is the only kernel in this project that encodes **which position in
the tour** each city occupies, not just which city is present — a full
N×N one-hot grid, so **N² qubits, not N**. Every other graph kernel here
(MaxCut, clique: N qubits; kcolor: N×k qubits) scales linearly in node
count. Confirmed directly, not just from the formula: `c2q_tsp` on a
6-city instance throws immediately —

```
threw as expected: TSP QAOA needs 36 qubits (limit 28). Reduce problem size.
```

— which means the entire testable range under the 28-qubit simulator
ceiling is N=3, 4, 5. Compare qubit cost at matched N:

| N | MaxCut/Clique (N) | KColor, m=3 (3N) | TSP (N²) |
|---|---|---|---|
| 3 | 3 | 9 | 9 |
| 4 | 4 | 12 | 16 |
| 5 | 5 | 15 | 25 |
| 6 | 6 | 18 | 36 — exceeds limit |

### Findings

- **N=3 (9 qubits, the cheapest tier): 4/10 correct, 6/10 invalid.**
  Already the worst cheapest-tier result of any kernel evaluated in this
  project — every other kernel's smallest tested size was near-perfect.
- **N=4 (16 qubits): 0/10 correct, 10/10 invalid.** Complete collapse,
  one qubit-tier below where clique/kcolor were still mostly reliable.
- **No partial credit observed.** `decode_tsp` only returns a real tour
  distance for a genuine valid permutation — same sound-when-it-answers
  property kcolor/clique's decoders have — but empirically, across all 20
  trials at N=3 and N=4, every single one was either *exactly* optimal or
  not a valid tour at all. Not claimed as structurally impossible (a
  valid-but-suboptimal tour is a representable outcome `decode_tsp` would
  correctly score), just not observed here — a QUBO landscape that looks
  closer to all-or-nothing than the "usually a close miss" shape clique
  and kcolor's harder cases showed.
- **Wall-clock cost outran what this GPU could finish, a full qubit-tier
  before the simulator's own 28-qubit ceiling would matter.** N=3 and N=4
  averaged 0.80s and 1.55s per seed respectively — cheap. N=5's single
  attempted seed exceeded 20 minutes without completing. For comparison,
  clique's and kcolor's own worst-case single-seed spot checks near their
  ceilings landed in the 1–4 minute range and *did* complete (see
  `analysis/clique/clique.md`) — TSP's N=5 didn't finish within roughly
  20× that budget.

### Practical limits

| Regime | N (cities) | Verdict |
|---|---|---|
| Marginal | 3 | 40% correct, 60% invalid — worst cheapest-tier result of any kernel tested |
| Failed | 4 | 0% correct, 100% invalid |
| Not completed | 5 | single seed exceeded 20 min on this GPU, aborted |
| Not reachable | ≥ 6 | 36+ qubits, exceeds the 28-qubit simulator limit outright |

### For the existing `tsp_impl` bridge

**Updated, per an explicit project decision to exercise the quantum path
within any range where it's verified correct, wall-clock cost aside**
(the same call made for `factor_impl` — see `analysis/factor/factor.md`).
`tsp_impl` now tries `c2q_tsp` first, but *only* when the total city
count is ≤3. That narrow gate matters for correctness, not just
performance: `decode_tsp` validates a returned tour is a genuine
Hamiltonian cycle, but validity alone doesn't prove it's the *minimum*-
cost one — at N=4+ there are multiple distinct valid tours with
different costs, so trusting a merely-valid kernel answer could silently
return a suboptimal result. At N≤3 a complete graph has exactly **one**
distinct Hamiltonian cycle (all edges, either direction, same total
either way), so valid and optimal necessarily coincide — a degenerate
property specific to N≤3 that must be re-derived, not assumed, before
ever raising this cutoff. A kernel "-1" (no valid tour found) still
falls through to the exact classical search, same
ambiguous-negative-never-trusted convention as `kcolor_impl`.

## 1a. A real `qubo_tsp` bug, found, fixed, and verified — but it doesn't rescue the empirical numbers

Prompted by how much worse TSP's collapse was than every other kernel's,
checked whether `qubo_tsp` (`c2cudaq/include/c2cudaq/internal.h`) itself
was constructing the right optimization target — pure classical
computation, no GPU needed at all
(`analysis/tsp/tsp_qubo_check.cpp`: brute-forces the QUBO's own true
minimum over every bitstring and compares it against the classical TSP
optimum).

**Found two real, confirmed bugs, both now fixed:**

1. **Missing wraparound edge.** The distance-cost term (`H_B`) only
   summed transitions `position p → position p+1` for `p` up to `n-2` —
   it never scored the tour's *closing* edge (`position n-1 → position
   0`), even though `decode_tsp` (which scores correctness) does close
   the loop. Confirmed directly: zero QUBO coupling existed between any
   `(v, position n-1)` and `(u, position 0)` pair.
2. **Half the distance-cost terms silently dropped.** `Q` is documented
   and consumed as upper-triangular only (`qubo_to_ising` never reads a
   cell where row > column) — but the old write
   `Q[(v*n+p)*dim + (u*n+p+1)]` isn't canonically ordered: whenever
   `v > u`, that index lands in the lower triangle, a cell nothing ever
   reads. Since the outer loop visits every ordered `(v, u)` edge pair,
   this silently discarded roughly *half* of the intended distance-cost
   signal, not just the wraparound piece. Confirmed by direct
   instrumentation on a 4-node graph: 18 nonzero entries landed in the
   dead lower triangle, matching the predicted count (6 `v>u` pairs × 3
   positions) exactly.

**Impact, measured precisely:** at N=4, the *old* QUBO's own true global
minimum (found by brute force, not by QAOA) was a **valid tour of length
48** — the actual optimum is **35**. Even a hypothetically perfect
optimizer converging exactly to the old QUBO's minimum would have
returned the wrong answer. Both bugs fixed in
`c2cudaq/include/c2cudaq/internal.h` (canonical `min/max` cell ordering
+ `(p+1) % n` wraparound, applied to both the distance-cost and
non-edges-penalty loops). Re-verified after the fix: the QUBO's own
minimum now exactly matches the classical optimum at both N=3 (44) and
N=4 (35).

**But re-running the actual QAOA stress test after the fix found no
improvement** — if anything, slightly worse on this small sample:

| N | Before fix | After fix |
|---|---|---|
| 3 (9 qubits) | 4/10 correct, 6/10 invalid | 1/10 correct, 9/10 invalid |
| 4 (16 qubits) | 0/10 correct, 10/10 invalid | 0/10 correct, 10/10 invalid |

This is the precise, useful negative result: the QUBO construction bug
was real and worth fixing (it made the *optimization target itself*
wrong, a genuine correctness defect, independent of QAOA), but it is
**not** what's causing the empirical completeness collapse. The
constraint-satisfaction terms (`H_A`, which enforce "one city per
position" / "one position per city") were already correct before this
fix — verified separately, they only ever write to the upper triangle
and were untouched by either bug. The collapse is a genuine QAOA
convergence/landscape difficulty on a now-provably-correctly-specified
problem, not an implementation defect masquerading as one. This
strengthens rather than weakens the case for keeping `tsp_impl`
classical-only: the kernel's difficulty here isn't a bug waiting to be
found.

## 2. A correction, discovered mid-stress-test: this project's kernel
   reports have all been running on GPU, not CPU

While debugging why the TSP N=5 point was taking so long, confirmed via
`nvidia-smi` (100% GPU utilization, 415MiB used, RTX 3050) and `nvq++`'s
own script logic that **`nvq++` silently defaults to the `nvidia` target
(cuStateVec, FP32, GPU-backed) whenever a CUDA GPU is present on the
machine — not `qpp-cpu` (CPU-backed, FP64) as every prior report in this
project assumed and stated.** `nvq++`'s compile script checks
`query_gpu` and, if a GPU + `libnvqir-custatevec-fp32.so` are found,
overrides `TARGET_CONFIG` from `qpp-cpu` to `nvidia` before any user
flag is considered.

This means `analysis/clique/report.html`, `analysis/factor/report.html`,
and the first draft of `analysis/maxcut/report.html` all mislabeled their
footer as "qpp-cpu statevector simulator." **Corrected in all three** —
footers now read "`nvidia` target (cuStateVec, FP32, GPU-backed)" with a
note on the original mislabeling. This doesn't change any reported
objective values or pass/fail outcomes (those come from `decode_*`
functions operating on the returned bitstring, backend-independent) — it
only corrects what hardware produced the wall-clock timings in every
report. For factor's report specifically, this is a point in favor of
its "not worth it" verdict, not against it: the quantum-vs-classical
timing comparison there already used the faster of the two backends and
still lost by 5–10 orders of magnitude.

## Next steps (when revisited)

1. ~~Stress-test `c2q_tsp`.~~ Done — see above.
1a. ~~Wire `c2q_tsp` into `tsp_impl` where it's verified safe.~~ Done —
   kernel-first at total city count ≤3 (see "For the existing `tsp_impl`
   bridge" above), classical fallback otherwise.
2. ~~Check for a `qubo_tsp` implementation bug.~~ Done — found and fixed
   two real bugs (§1a: missing wraparound edge, half the distance-cost
   terms silently dropped from the upper-triangular storage). Confirmed
   the fix makes the QUBO's own true minimum match the classical optimum
   at N=3/N=4. Confirmed the fix does *not* rescue empirical QAOA hit
   rates — the collapse is a genuine convergence problem on a
   provably-correct landscape, not a masked bug. Fix is live in
   `c2cudaq/include/c2cudaq/internal.h`; `libc2cudaq.a` rebuilt and
   verified against the other four kernels' integration tests (no
   regressions — `qubo_tsp` is TSP-only, not shared).
3. If ever revisited on hardware that can actually complete an N=5 run
   (a workstation-class or datacenter GPU, or simply more patience than
   this stress test had budget for), fill in the N=5 data point properly
   rather than leaving it as "attempted, not completed" — now against
   the corrected kernel.
4. The GPU-vs-CPU backend question (§2 above) is now resolved for this
   project, but worth remembering if `nvq++` is ever invoked with an
   explicit `--target=qpp-cpu` for a future comparison — that would be a
   genuinely different (slower, FP64) baseline than everything measured
   so far.
5. ~~Fix `test/integration/run_*.sh` / `tools/qoffload-clang++` hardcoding
   the CPU backend.~~ Done. Found while re-running the integration tests
   against the new kernel-first bridges: unlike raw `nvq++` (which
   auto-detects the GPU per §2's finding), all five `run_*_integration.sh`
   scripts and `tools/qoffload-clang++` hardcoded `-lnvqir-qpp` (CPU)
   unconditionally at their link step — meaning the actual pass-generated
   pipeline could never reproduce the GPU wall-clock numbers documented
   in any of this project's four `.md` reports, no matter what hardware
   ran it. Confirmed concretely: `factor_e2e` (n=91, kernel-first after
   the bridge change above) took 46.9s once fixed, matching
   `factor.md`'s GPU number exactly — vs. 5+ minutes and still running
   before the fix. All six scripts now replicate `nvq++`'s own
   `query_gpu()` check and pick `-lnvqir-custatevec-fp32` when a GPU is
   present, `-lnvqir-qpp` otherwise.
