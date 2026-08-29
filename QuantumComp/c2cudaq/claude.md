# CLAUDE.md — SQIF Factorization Path (new API, additive alongside grover.cpp)

## Context

`src/grover.cpp` implements `factor_grover_kernel`, a Grover's-algorithm
factoring oracle (QFT-multiply → phase oracle → inverse QFT-multiply →
diffuser). It's search-based, so qubit/gate count scales with the bit-length
of `n` in a way that caps us around `n < 128` on the current GPU.

We're adding an alternative factoring path based on **SQIF** (Sublinear
Quantum Integer Factorization), from Yan et al., "Factoring integers with
sublinear resources on a superconducting quantum processor" (arXiv:2212.12372,
`baoetal.pdf` in repo root — read it directly, especially Supplementary
Material sections I–VI, which has exact worked examples to validate against).

SQIF is a hybrid classical+quantum algorithm: classical lattice reduction
(LLL + Babai's algorithm) sets up a small optimization problem, and QAOA
refines it. Qubit count scales as `n ≈ 2c·log(N)/log(log(N))` — sublinear —
so factoring larger `N` costs *more classical work and more sr-pair
collection iterations*, not more qubits. This is a fundamentally different
resource curve from Grover's search, which is why it's worth adding as a
second path past the current 128 ceiling.

**This is a new, permanent, additive API — not a replacement.**
`grover.cpp`, its tests, and `c2q_factor(n)` stay exactly as they are and
keep being maintained. SQIF ships as its own function
(`c2q_factor_sqif(n)` — see Integration below) that callers pick explicitly;
there is no dispatcher, mode flag, or env var that silently swaps one for
the other. `shor.cpp` is dead experimental code — leave it alone, don't
build on it, don't delete it unless asked.

## Step 0 — verified facts about the existing codebase

**A previous pass through this file made several assumptions about the
codebase that turned out to be wrong when actually checked. Corrected below
so we don't re-derive them.**

- **`src/qaoa.cpp`** and **`src/qubo.cpp`** — as suspected, this repo already
  has generic QAOA-kernel and QUBO/Ising infrastructure, and it's a strong
  fit for Stage 2:
  - `qaoa_general` (the `__qpu__` kernel), `run_qaoa()` (COBYLA optimizer +
    `cudaq::observe`/`cudaq::sample` loop), and the `IsingTerms` struct
    (`zz_i/zz_j/zz_c`, `z_i/z_c`, `offset`) in `include/c2cudaq/internal.h`
    are fully generic — reuse them as-is. Build an `IsingTerms` directly
    from the SQIF distance-squared objective and call
    `run_qaoa(n_qubits, layers, ising, seed)`.
  - **Do not** route through `qubo_to_ising()` — that converts a `{0,1}`
    QUBO matrix, but SQIF's problem Hamiltonian (paper Eq. S47/S48) is
    already naturally in Ising form with `x_i ∈ {-1,0,+1}` encoded directly
    to Pauli-Z. Constructing `IsingTerms` by hand from the derived
    coefficients is the right fit, not going through the QUBO layer.
  - `run_qaoa` is currently `static` in `qaoa.cpp` — it'll need to become
    non-static (or moved somewhere shared) to call it from a new
    `sqif_qaoa.cpp` translation unit.
- **`src/arith.cpp`** — checked. It does **not** have gcd / modular
  exponentiation / number-theory helpers of any kind — it contains only
  `__qpu__` quantum add/sub/mul kernels. `shor.cpp` has file-local `static
  shor_gcd`/`shor_modpow` helpers, but they're unexported, `int64_t`-only,
  and `shor.cpp` is dead code we're not building on — don't reuse them as-is
  (see bignum note below for why `int64_t` isn't enough anyway). Write new
  gcd/modpow helpers for Stage 3, sized for the integer type actually in use
  there (see next point).
- **Arbitrary-precision integers are a hard requirement, not a
  "large N" edge case.** Confirmed by reading Section VI of the supplement:
  even the **5-qubit** worked example (`N = 48567227`, only 26 bits) has
  final `X`/`Y` values (Eq. S64) that are ~200-digit numbers, because
  `X = Π u_j^{t_j}` multiplies across ~20–30 smooth relation pairs and each
  `u_j` is already large — the product blows up combinatorially independent
  of how big `N` itself is. `int64_t` (this repo's only integer type
  anywhere) and even `__int128` are nowhere close to sufficient once you
  combine more than a couple of sr-pairs.
  - The repo currently has **zero** big-integer support (checked — no gmp,
    boost::multiprecision, or hand-rolled bignum anywhere).
  - **Use GMP.** It's installed on this machine (`gmp.h`/`gmpxx.h` headers
    and `libgmp`/`libgmpxx` both present). Confirmed by direct smoke test
    that `mpz_class` arithmetic (construct/multiply/`get_str()`) compiles
    and links fine through `nvq++`.
  - **Caveat:** `gmpxx`'s convenience `operator<<(std::ostream&, ...)`
    does **not** link through `nvq++` — undefined symbol, because
    `libgmpxx.so` was built against libstdc++'s `std::ostream` ABI while
    `nvq++`/clang uses libc++, and the manglings don't match. Use
    `.get_str()` (or a hand-written wrapper around it) instead of streaming
    `mpz_class`/`mpq_class` directly to `std::cout`/`std::ostringstream`
    anywhere in this codebase.
  - Use `mpq_class` (exact rational) for Gram-Schmidt/LLL in Stage 1 rather
    than floating point — see the LLL-uniqueness note under Test plan for
    why exactness matters here.
  - `CMakeLists.txt` has zero `find_package`/`find_library` calls today.
    Adding GMP means adding that wiring (`find_library(GMP_LIB gmp)`,
    `find_library(GMPXX_LIB gmpxx)`, link into the `c2cudaq` target) —
    it doesn't exist yet.
- **There is no per-file compiler split — everything compiles through
  `nvq++` uniformly.** Checked `CMakeLists.txt`: `nvq++` is set as the
  single project-wide `CMAKE_CXX_COMPILER`, and *every* `.cpp` file —
  library, tests, examples, including the already-pure-host `qubo.cpp` —
  compiles through it with no distinction between "kernel-bearing" and
  "plain host" files. New pure-host Stage 1/3 files (`sqif_lattice.cpp`,
  `sqif_postprocess.cpp`) need no special CMake treatment: just add them to
  the same `add_library(c2cudaq ...)` source list as everything else.
- **`nvqpp_wrap.sh`** — dead file. Grepped the whole build; it's not
  referenced by any `CMakeLists.txt` anywhere. Ignore it — the actual fix
  for nvq++'s missing `-MD/-MF` dependency-file support lives directly in
  the root `CMakeLists.txt` now (`CMAKE_DEPENDS_USE_COMPILER FALSE` +
  a custom `CMAKE_CXX_COMPILE_OBJECT` rule, set before `project()`).
  (`README.md`'s build instructions still reference the old wrapper —
  that's pre-existing doc-rot, unrelated to SQIF, not worth fixing here.)
- **`tests/test_factor_bench.ncpp`** — the `.ncpp` extension isn't
  meaningful to any tool (nothing globs `.cpp`/`.ncpp`, sources are listed
  explicitly in `tests/CMakeLists.txt`, and `nvqpp_wrap.sh` — the one thing
  that might have cared — isn't invoked at all). It's simply omitted from
  the test target's source list, nothing more. Match convention for new
  SQIF test files by listing them explicitly in `tests/CMakeLists.txt`.
- **`src/bridge.cpp`** is *not* a general "how kernels get exposed" pattern
  — it's a narrow one-off `extern "C"` adapter for the separate LLVM
  maxcut-cpp-pass, unrelated to factoring. The actual pattern for exposing
  a new public entry point is simpler: implement it in a `.cpp`, declare it
  in `include/c2cudaq.h`. No bridge-style adapter needed for
  `c2q_factor_sqif`.

## Algorithm mapping to this codebase

### Stage 1 — Classical preprocessing (pure host C++, no `__qpu__`)

New file, e.g. `src/sqif_lattice.cpp` + a header exposing the API
(`include/c2cudaq/` or wherever fits — no existing precedent to match since
`arith.cpp` has no classical-helper header split; use your judgement, e.g. a
new `include/c2cudaq/sqif.h`):

1. Pick lattice dimension `n` and precision parameter `c`. The paper's own
   formula (`n ≈ 2c·logN/loglogN`, Sec. III) uses ad hoc rounding at each
   step and isn't a single clean reproducible function — for the three
   validated worked examples the paper states `n=3,c=1.5` / `n=5,c=4` /
   `n=10,c=4` directly rather than deriving them mechanically. **Hardcode
   `(n, c)` for the three validated test cases**; treat any general
   `n(N)`-formula for other `N` as a separate, lower-confidence
   extrapolation (see "Things to flag back").
2. Prime basis: first `n` primes, plus a sign placeholder `p0 = -1`. **This
   is a different, much smaller prime basis than Stage 3's** — see the
   disambiguation note in Stage 3 below. Don't reuse this list there.
3. Build lattice basis matrix `B` (`(n+1)×n`) and target vector `t`,
   following the paper's *integer* construction (Eq. S33/S34, not the more
   general real-valued Eq. S7): diagonal entries are `⌈i/2⌋` for
   `i=1..n`, randomly permuted (function `f`); bottom row is
   `⌈10^c · ln(p_i)⌋` for each prime in the basis; `t` is zero except a
   bottom entry `⌈10^c · ln N⌋`. **`⌈x⌋` here is round-to-nearest, not
   ceiling** — caught this by hand-verifying against Eq. S35's actual
   values: `10^4·ln(2) = 6931.47` rounds to `6931` (matches the paper) but
   *ceils* to `6932` (doesn't match). `std::ceil`/`std::lround` look
   similar enough to typo one for the other — use round. (The diagonal
   `⌈i/2⌋` values happen to come out identical either way, since `i/2` is
   always an exact integer or exact half-integer, so this bug is only
   visible in the weight row / target, not the diagonal — easy to test one
   and think both are right.) Verified this exact construction against
   the paper's own `B_{3,1.5}`/`t_3` (Eq. S37), `B_{5,4}`/`t_5`
   (Eq. S35/S36), `B_{10,4}`/`t_10` (Eq. S38/S39) — reproduce these three
   matrices exactly as unit tests; they're small enough to hardcode as
   expected output.
4. **LLL reduction** (`δ = 3/4`) on `B` → reduced basis `D`. Standard
   textbook algorithm (Gram-Schmidt → size-reduce → swap loop). **Use exact
   rational arithmetic (GMP `mpq_class`) for the Gram-Schmidt coefficients,
   not floating point** — see the Test plan note on why LLL output isn't
   unique and how that interacts with regression testing.
5. **Babai's nearest-plane algorithm** on `D`, `t` → approximate closest
   vector `b_op`, plus the real-valued Gram-Schmidt coefficients `μ_i` and
   their rounded integers `c_i` (needed for Stage 2's qubit encoding).

Nothing in this stage touches CUDA-Q. It should be fully unit-testable
without a GPU.

### Stage 2 — QAOA refinement (this is the `__qpu__` part)

New file, e.g. `src/sqif_qaoa.cpp`:

1. **Encode floating coefficients.** For each `i`, `x_i ∈ {-1, 0, +1}` floats
   around Babai's `c_i`. Map to qubit `i` per the sign of `μ_i − c_i`
   (paper Eq. S48, verified): if `c_i` was rounded down (`c_i ≤ μ_i`),
   `|0⟩/|1⟩ → x_i = 0/+1`; if rounded up (`c_i > μ_i`),
   `|0⟩/|1⟩ → x_i = 0/−1`. Compute this per-qubit sign at preprocessing
   time from Stage 1's output — it's data-dependent, not fixed.

2. **Build the problem Hamiltonian directly as `IsingTerms`** (see Step 0 —
   *don't* go through `qubo_to_ising`). Substituting `x_i → Z_i` into
   `||t − Σ x_i·d_i − b_op||²` gives a fully-connected Ising Hamiltonian:
   a constant, linear `Z_i` terms, and `Z_i Z_j` terms for every pair
   (verified: this is exactly `K_n`, paper's own description). No need for
   the paper's SWAP-network routing (that was for their real 1D-chain
   hardware) — for simulation, apply the `K_n` Hamiltonian directly, same
   as `qaoa_general` already does for MaxCut/etc. Cross-check the generated
   coefficients against the paper's fully expanded Hamiltonians — verified
   exact match available at Eq. S51 (3-qubit), Eq. S50 (5-qubit), Eq. S52
   (10-qubit) — use these as regression tests, they're small closed-form
   expressions.

3. **Kernel structure**: reuse `qaoa_general` from `qaoa.cpp` unmodified —
   it already takes `IsingTerms`-shaped inputs
   (`n, p, zz_i, zz_j, zz_c, z_i, z_c, thetas`) and needs no SQIF-specific
   changes. Only write new kernel code if `IsingTerms` genuinely can't
   express something SQIF needs (unlikely — the Hamiltonian is plain
   Z/ZZ, same shape as MaxCut's).

4. **Classical optimization loop** over `(γ, β)` minimizing `⟨H_c⟩` via
   `cudaq::observe`. Reuse `run_qaoa()` from `qaoa.cpp` (make it non-static
   so `sqif_qaoa.cpp` can call it) rather than writing a second copy — it
   already does COBYLA + observe/sample and returns
   `{bitstring, energy}`. The paper's own Model Gradient Descent
   (Supplement Algorithm 3) is not required — confirmed the paper states
   MGD converges within ~10 steps at these small qubit counts, and COBYLA
   is a reasonable substitute; no need to implement MGD specifically.

5. **Sample and decode.** Take the top few bitstrings from `run_qaoa`'s
   `cudaq::sample` call, decode to `(x_1,...,x_n)`, compute
   `v_new = b_op + Σ x_i·d_i`, then `(u, v)` via
   `u = Π p_i^{e_i}` (positive exponents), `v = Π p_i^{-e_i}` (negative
   exponents) — paper Eq. S8. **`u`/`v` themselves can already exceed
   `int64_t`** for larger cases — use `mpz_class` here too, not just in
   Stage 3.

6. **Smoothness check.** Test `|u − v·N|` smooth over **Stage 3's prime
   basis** (not Stage 1's — see disambiguation below) up to bound `B2`.
   Verified exact `(B1, B2)` values for all three cases in Table S5:
   3-qubit `B1=5, B2=47`; 5-qubit `B1=11, B2=229`; 10-qubit `B1=29,
   B2=1223`. Use these as hardcoded expected bounds for the three
   validated cases rather than re-deriving `B2` from a formula.

7. **Loop.** Re-run Stage 1 with a different random diagonal permutation
   (paper confirms this — Sec. IV.A: "a random permutation function `f` is
   used to perform random permutation on the diagonal elements") until
   enough independent sr-pairs are collected. Paper collects 20/55/221
   pairs for the 3/5/10-qubit cases respectively (their exact lists are in
   Section VI — usable directly as fixture data, see Test plan).

### Stage 3 — Postprocessing (pure host C++)

New file, e.g. `src/sqif_postprocess.cpp`:

**Important disambiguation (not clear in the original draft of this plan):
Stage 3 uses a *different, larger* prime basis than Stage 1.** Stage 1's
prime basis is just the first `n` primes (e.g. `{2,3,5,7,11}` for the
5-qubit case, `n=5` primes) — used only to build the lattice matrix. Stage
3's prime basis is **all primes ≤ `B2`** — e.g. 50 primes ≤ 229 for the same
5-qubit case (verified against Table S5's `B2-dim` column: 15/50/200 primes
for the 3/5/10-qubit cases, each exactly equal to the count of primes below
that case's `B2`). The GF(2) exponent matrix's columns range over *this*
basis, not Stage 1's. `eq-dim` in Table S5 (16/51/201) is `B2-dim + 1` for
the sign bit — that's the matrix column count and the dimension you're
solving the linear system in.

1. Build the Boolean exponent matrix (rows = sr-pairs, columns = the
   `B2`-bounded prime basis above, including the `p0=-1` sign bit).
2. `GF(2)` Gaussian elimination to find a linearly dependent subset
   (coefficients `t_j ∈ {0,1}` s.t. combined exponents are all even —
   paper Eq. S5). Matrix sizes are small (up to 221×201 for the 10-qubit
   case) — plain bitset-based Gaussian elimination is more than fast
   enough, no special sparse solver needed.
3. `X = Π u_j^{t_j}`, `Y = sqrt(Π (u_j − v_j·N)^{t_j})`. **Must use
   `mpz_class`** — verified the paper's own 5-qubit worked example produces
   ~200-digit `X`/`Y` values (Eq. S64), and this isn't a rare case, it's the
   norm once several sr-pairs combine. Use the paper's fully worked numeric
   examples (Section VI: full sr-pair lists, Eq. S64/S66/S69 for the
   solution vectors, Eq. S68/S71 for the final gcd results) as golden
   regression tests for this step — they pin down exact big-integer values,
   which is exactly where subtle bugs hide.
4. `p = gcd(X+Y, N)`, `q = gcd(X−Y, N)`. If `X ≡ ±Y mod N` (trivial), discard
   this dependent subset and try another, or collect more sr-pairs. Write
   a new `mpz_class`-based gcd (Euclidean algorithm, trivial to implement,
   or GMP's own `mpz_gcd` — no need to hand-roll it) — do not try to adapt
   `shor.cpp`'s `int64_t` gcd, it's the wrong integer type for this stage.

## Integration

- New public entry point `c2q_factor_sqif(int64_t n)` (matching
  `c2q_factor`'s naming pattern) declared in `include/c2cudaq.h`, returning
  the same `std::pair<int64_t, int64_t>` shape (final factors fit in
  `int64_t` even though intermediate postprocessing values don't — cast
  down only at the very end, after `gcd`). Implemented via `src/sqif.cpp`
  orchestrating Stages 1–3 and the sr-pair collection loop.
- **No path-selector/dispatcher needed.** Both `c2q_factor` (Grover) and
  `c2q_factor_sqif` (SQIF) are kept permanently as independent public
  functions — callers choose which to call directly. Update
  `examples/example_factor.cpp` to demonstrate both, but don't build a
  flag/env-var/constructor-param switch between them.
- Update `CMakeLists.txt` (root only — `tests/`/`examples/` just need new
  `add_c2_test`/`add_c2_example` lines, no generator changes) to build the
  new files and link GMP. Since there's no per-file compiler split (see
  Step 0), every new file — kernel-bearing or pure host — just gets added
  to the existing `add_library(c2cudaq ...)` source list.
- Do not modify `grover.cpp`, its tests, or `factor_01_grover.qasm` /
  `factor_01.json` (those look like fixed reference artifacts for the
  Grover path — leave them as-is unless asked).

## Test plan — paper's own worked examples as ground truth

Add `tests/test_sqif.cpp`, reproducing, in order:

1. `N = 1961` (11-bit, `n=3`, `c=1.5`) → expect `37 × 53`. Assert against
   the exact lattice matrix (Eq. S37) and Hamiltonian coefficients
   (Eq. S51) exactly — these are small closed-form values with no
   ambiguity. For the LLL-reduced basis and `b_op`, see the note below
   before asserting exact equality.
2. `N = 48567227` (26-bit, `n=5`, `c=4`) → expect `7919 × 6133`. Same
   pattern: Eq. S35/S36 (lattice), Eq. S50 (Hamiltonian) exact; Eq. S41/S43
   (LLL/`b_op`) per the note below.
3. `N = 261980999226229` (48-bit, `n=10`, `c=4`) → expect
   `15538213 × 16860433`. Same pattern with Eq. S38/S39, S52, and S42/S45.

**Note on LLL-reduced-basis (`D`) and `b_op` regression tests:** LLL
reduction is not unique — tie-breaking in the swap step and
floating-point-vs-exact-rational Gram-Schmidt can both legitimately produce
a different, equally-valid `δ=3/4`-reduced basis than the paper's own
`D_{3,1.5}`/`D_{5,4}`/`D_{10,4}` (Eq. S40/S41/S42) without the
implementation being wrong. Don't require byte-identical `D` matrices as a
hard test failure. Instead:
- Test the *properties* that must hold for any valid LLL-reduced basis
  (size-reduction and Lovász conditions from Algorithm 1 in the paper).
- Test `b_op` and the derived Hamiltonian coefficients against the paper's
  values as the real regression target — these are far more likely to
  match exactly (Babai's rounding is comparatively directly determined
  once you have *a* valid reduced basis of the right quality), and they're
  what actually matters for correctness. If using exact `mpq_class`
  arithmetic reproduces the paper's own `D` matrices bit-for-bit, great,
  assert on that too — just don't treat a mismatch there alone as a bug if
  `b_op`/Hamiltonian/energy-spectrum still match.

Only after all three pass, try `N` beyond the Grover oracle's current 128
ceiling — that's the actual goal of this new path. Track whether the
bottleneck becomes classical (LLL/Babai/GF(2)-solve/big-int arithmetic)
rather than qubit count, since that's the expected failure mode as `N`
grows, not a qubit-count wall like `grover.cpp` hits.

## Things to flag back rather than silently resolve

- `c` and the smoothness bound `B2` are hand-tuned per case in the paper,
  not formula-derived — confirmed `n(N)`'s own formula involves ad hoc
  rounding even in the paper's own worked example (`N=48567227`: bit-length
  26, `loglogN` stated as 5, `n=26/5≈5` — not a single clean function). If
  convergence is poor on a new `N`, or the general dimension formula
  disagrees with a hardcoded value on one of the three validated cases, say
  so and propose a sweep rather than guessing a value and moving on.
  **Confirmed this happens in practice, not just hypothetically:**
  `c2q_factor_sqif(8633)` (14-bit, unvalidated fallback path) fails.
  Diagnosed two distinct problems by instrumenting the pipeline directly
  (see git history around the commit that added this note for the
  standalone repro, not kept in-tree):
  1. The fallback's flat `c=4.0` (borrowed from the 26-/48-bit validated
     cases) is badly miscalibrated for a 14-bit `N` — it produced `u`/`v`
     values up to ~10^18 (bigger than `N` itself) with zero B2=50-smooth
     hits across 40 rounds. Dropping to `c=1.5` (the 11-bit case's value)
     immediately produced smooth sr-pairs. `c` needs to scale with `N`'s
     bit-length, not be a flat constant past the three validated cases.
  2. Even after fixing `c`, both `n=4` and `n=5` converged to only **two**
     distinct smooth `(u,v)` pairs across 60 rounds / thousands of sampled
     bitstrings — every found GF(2) dependency turned out to be a
     duplicate-pair self-square (`u_i == u_j` for the two indices
     combined), hence always trivial (`X ≡ Y` exactly, not just mod `N`).
     This means the paper's prescribed randomization (permute the
     diagonal, Sec. IV.A) doesn't inject enough diversity into the CVP
     solution at these small lattice dimensions for this particular `N` —
     Babai converges to essentially the same closest vector regardless of
     which of the ~6 (n=4) or ~30 (n=5) diagonal permutations is used.
     Bumping QAOA layers (`p=1→3`) did not help either, confirming this
     isn't a QAOA-sampling-entropy problem but a Stage-1 diversity problem.
     Untried next steps: sampling genuinely random diagonals (not just
     permutations of the fixed `⌈i/2⌋` multiset — a deviation from the
     paper's exact method), varying `c` across rounds too, or accepting
     that `n` may need to grow faster with `N` than the sublinear formula
     suggests for `N` in this size range. Don't silently paper over a
     repeat of this with a bigger `max_rounds` or `target_pairs` alone —
     the diagnostic above shows more rounds just re-finds the same two
     pairs, it doesn't discover new ones.
- The paper's own authors note QAOA's advantage over plain classical Babai's
  (no quantum step at all) is inconclusive at these scales (Supplement Sec.
  VII, Fig. S4). If a classical-only Babai's-with-resampling baseline
  matches or beats the QAOA path on our sizes, report that as a real finding
  — don't tune parameters until QAOA looks better than it is.
- If reusing `qaoa_general`/`run_qaoa` genuinely doesn't fit once you're
  writing Stage 2 for real (e.g. the per-qubit encoding needs something
  `IsingTerms` can't express), say what's reusable vs. what needs to
  diverge, rather than either forcing a bad fit or duplicating silently.
- If GMP's `mpz_class`/`mpq_class` turns out to have other libc++-ABI link
  issues beyond the known `operator<<` one (found via one smoke test, not
  an exhaustive audit), flag it rather than working around it silently —
  it may indicate a broader pattern worth handling once at the CMake level
  (e.g. an isolated wrapper translation unit) rather than per call-site.
