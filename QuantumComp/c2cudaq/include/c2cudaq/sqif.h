#pragma once
// SQIF (Sublinear Quantum Integer Factorization) internals.
// Yan et al., "Factoring integers with sublinear resources on a
// superconducting quantum processor" (arXiv:2212.12372). See claude.md for
// the stage-by-stage mapping and paper equation references.
//
// Pure host arithmetic uses GMP (mpz_class/mpq_class). Do not stream
// mpz_class/mpq_class via operator<< anywhere in this codebase -- it does
// not link through nvq++ (libc++ vs. libstdc++ ABI mismatch in libgmpxx).
// Use .get_str() instead.
#include <c2cudaq/internal.h>
#include <gmpxx.h>
#include <optional>
#include <string>
#include <vector>

namespace c2cudaq {

// ---- Stage 1: lattice construction + LLL + Babai --------------------------

// Basis vectors are stored one-per-entry: basis[i] is the i-th column
// vector of the paper's B_{n,c} matrix, dimension n+1 (first n components =
// diagonal/off-diagonal block, last component = the "weight" row).
using SqifBasis = std::vector<std::vector<mpz_class>>;

struct SqifLattice {
    int n = 0;
    SqifBasis B;                 // n vectors, each of dimension n+1
    std::vector<mpz_class> t;    // target vector, dimension n+1
    std::vector<int> primes;     // first n primes (Stage-1 basis only)
};

// diag[i] must be a permutation of the multiset {ceil(1/2),...,ceil(n/2)}.
SqifLattice sqif_build_lattice(int64_t N, int n, double c,
                                const std::vector<long>& diag);

// Identity-order permutation: diag[i] = ceil((i+1)/2).
std::vector<long> sqif_default_diag(int n);

// A pseudo-random permutation of sqif_default_diag(n), seeded, used to draw
// a fresh CVP sample each sr-pair-collection round (paper Sec. IV.A).
std::vector<long> sqif_random_diag(int n, unsigned seed);

// LLL-reduce (delta = 3/4) the n column vectors of B.
SqifBasis sqif_lll_reduce(const SqifBasis& B, double delta = 0.75);

struct BabaiStep {
    mpq_class mu;   // real-valued Gram-Schmidt coefficient
    mpz_class c;    // mu rounded to nearest integer
};

struct BabaiResult {
    std::vector<mpz_class> b_op;   // dimension n+1: approximate closest vector
    std::vector<BabaiStep> steps;  // n entries, steps[i] aligned to D[i] (column i)
};

BabaiResult sqif_babai(const SqifBasis& D, const std::vector<mpz_class>& t);

// ---- Stage 2: Hamiltonian construction -------------------------------------

// signs[i] = +1 if x_i in {0,+1} (c_i <= mu_i, rounded down), -1 if
// x_i in {0,-1} (c_i > mu_i, rounded up). Paper Eq. S48.
std::vector<int> sqif_encoding_signs(const BabaiResult& babai);

// Builds Hc = ||t - sum x_i d_i - b_op||^2 as an Ising Hamiltonian, with
// x_i substituted by the operator (signs[i]/2)(I - Z_i). Exact-rational
// internally (matches paper Eq. S50-S52 to machine precision on get_d()).
IsingTerms sqif_build_hamiltonian(const SqifBasis& D,
                                   const std::vector<mpz_class>& t,
                                   const std::vector<mpz_class>& b_op,
                                   const std::vector<int>& signs);

// ---- Stage 3: sr-pair extraction + postprocessing --------------------------

// Stage 1's small basis (first n primes) -- NOT the same as the big_primes
// basis below. Used only to decode a QAOA bitstring into (u, v) via Eq. S8.
std::vector<int> sqif_first_n_primes(int n);

// Stage 3's basis: all primes <= B2. This is what the GF(2) exponent
// vectors below are indexed over (index 0 reserved for the p0=-1 sign
// bit), and is much larger than sqif_first_n_primes for the same n -- see
// claude.md's prime-basis disambiguation note.
std::vector<int> sqif_primes_upto(int B2);

// Decodes a QAOA measurement bitstring (qubit i = bit i, matching
// qaoa_general's convention) into x_i in {0, signs[i]}, computes
// v_new = b_op + sum x_i * D[i], then (u, v) via Eq. S8 using
// small_primes, plus u's own signed exponent vector over small_primes
// (u_exponents[i] = max(v_new[i], 0)) -- needed by sqif_try_sr_pair below,
// see the note on SrPair for why. Returns false if v_new's first n
// components aren't representable (shouldn't happen by construction;
// guards decode bugs).
bool sqif_decode_uv(const std::string& bits, const SqifBasis& D,
                     const std::vector<mpz_class>& b_op,
                     const std::vector<int>& signs,
                     const std::vector<int>& small_primes,
                     mpz_class& u, mpz_class& v,
                     std::vector<int>& u_exponents);

// IMPORTANT (found by hand-verifying paper Eq. S57/S61, the "10th + 17th
// sr-pair" worked example): the final X is sqrt(prod u_j over the chosen
// subset), NOT prod u_j itself -- that only works out because Schnorr's
// method requires prod u_j to already be a perfect square. That in turn
// means u_j's own exponents (over the *small*, Stage-1 prime basis) must
// ALSO sum to even across the subset, exactly like d_j's exponents (over
// the *big*, B2-bounded basis) already must. These are two independent
// GF(2) constraint systems that both have to be satisfied by the same
// subset -- so both exponent vectors get stored and concatenated into one
// combined system below, not just d_exponents alone.
struct SrPair {
    mpz_class u, v;
    // u's exponents over Stage 1's small (first-n-primes) basis -- from
    // sqif_decode_uv, always >= 0 (u is built only from positive-exponent
    // primes by construction).
    std::vector<int> u_exponents;
    // Signed exponents of u - v*N over {p0=-1} union big_primes: index 0
    // is the sign bit (1 if u - v*N < 0), index 1+k is the exponent of
    // big_primes[k]. Length == big_primes.size() + 1.
    std::vector<int> d_exponents;
};

// Checks gcd(u,v)==1 and that |u - v*N| is B2-smooth over big_primes.
std::optional<SrPair> sqif_try_sr_pair(const mpz_class& u, const mpz_class& v,
                                        const std::vector<int>& u_exponents,
                                        int64_t N,
                                        const std::vector<int>& big_primes);

// GF(2) Gaussian elimination over the *combined* [u_exponents |
// d_exponents] parity vector of each sr-pair (small_dim + big_dim
// columns). Returns every independent dependent subset found (t_j in
// {0,1} s.t. both u_j's and d_j's combined exponents are all even) -- not
// just the first -- so the caller can retry with a different subset if
// one turns out trivial (X = +-Y mod N).
std::vector<std::vector<int>> sqif_gf2_dependencies(
    const std::vector<SrPair>& pairs, int small_dim, int big_dim);

// X = sqrt(prod u_j), Y = sqrt(prod|u_j - v_j*N|) over the given subset,
// then p = gcd(X+Y,N), q = gcd(X-Y,N). Returns {1,1} if the dependency is
// trivial (X = +-Y mod N). Throws if the subset doesn't actually satisfy
// both perfect-square constraints (a bug upstream, not a normal runtime
// condition -- sqif_gf2_dependencies should never return such a subset).
std::pair<mpz_class, mpz_class> sqif_extract_factors(
    const std::vector<SrPair>& pairs, const std::vector<int>& subset,
    int64_t N);

}  // namespace c2cudaq
