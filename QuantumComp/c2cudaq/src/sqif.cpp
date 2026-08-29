#include <c2cudaq.h>
#include <c2cudaq/sqif.h>
#include <cudaq.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

using namespace c2cudaq;

namespace {

struct SqifParams {
    int n;
    double c;
    int B2;
    int target_pairs;
};

// Hardcoded for the three cases validated against the paper's own worked
// examples (arXiv:2212.12372 Sec. IV, Table S5 -- see claude.md). Anything
// else falls back to the paper's approximate sublinear dimension formula,
// which is lower-confidence: the paper itself derives n via ad hoc
// rounding on its own worked examples rather than a single clean formula.
SqifParams sqif_pick_params(int64_t N) {
    if (N == 1961) return {3, 1.5, 47, 20};
    if (N == 48567227) return {5, 4.0, 229, 55};
    if (N == 261980999226229LL) return {10, 4.0, 1223, 221};

    double logN = std::log2((double)N);
    double loglogN = std::max(std::log2(std::max(logN, 2.0)), 1.0);
    int n = std::max(3, (int)std::round(logN / loglogN));
    double c = 4.0;
    // B2 grows with the count of validated cases' B2-dim (15/50/200 primes
    // for n=3/5/10); no validated formula exists for arbitrary n, so this
    // is a rough extrapolation only -- see claude.md's "things to flag".
    int B2 = 50 + 150 * std::max(0, n - 5);
    return {n, c, B2, 8 * n};
}

}  // namespace

std::pair<int64_t, int64_t> c2q_factor_sqif(int64_t N) {
    if (N < 4)
        throw std::invalid_argument("c2q_factor_sqif: n must be >= 4");
    if (N % 2 == 0) return {2, N / 2};

    SqifParams params = sqif_pick_params(N);
    int n = params.n;
    check_qubit_limit(n, "SQIF QAOA");

    auto small_primes = sqif_first_n_primes(n);
    auto big_primes = sqif_primes_upto(params.B2);
    int basis_dim = (int)big_primes.size() + 1;

    std::vector<SrPair> pairs;
    std::mt19937 seed_rng(13);

    const int max_rounds = 300;
    for (int round = 0;
         round < max_rounds && (int)pairs.size() < params.target_pairs + 4;
         ++round) {
        auto diag = sqif_random_diag(n, seed_rng());
        auto lattice = sqif_build_lattice(N, n, params.c, diag);
        auto D = sqif_lll_reduce(lattice.B);
        auto babai = sqif_babai(D, lattice.t);
        auto signs = sqif_encoding_signs(babai);
        auto ising = sqif_build_hamiltonian(D, lattice.t, babai.b_op, signs);

        auto [top_bits, energy, opt_par] =
            run_qaoa(n, /*layers=*/1, ising, /*seed=*/round);
        (void)top_bits;
        (void)energy;

        auto counts = sqif_sample_qaoa(n, /*layers=*/1, ising, opt_par);
        for (auto& [bits, count] : counts) {
            (void)count;
            mpz_class u, v;
            std::vector<int> u_exponents;
            if (!sqif_decode_uv(bits, D, babai.b_op, signs, small_primes, u, v,
                                 u_exponents))
                continue;
            if (u <= 1 && v <= 1) continue;  // trivial (x = 0 everywhere)
            if (auto sr = sqif_try_sr_pair(u, v, u_exponents, N, big_primes))
                pairs.push_back(*sr);
        }
    }

    if (pairs.empty()) return {1, N};

    auto deps = sqif_gf2_dependencies(pairs, n, basis_dim);
    for (auto& subset : deps) {
        auto [p, q] = sqif_extract_factors(pairs, subset, N);
        if (p > 1 && p < N && (N % p.get_si()) == 0) {
            int64_t pi = p.get_si();
            return {std::min(pi, N / pi), std::max(pi, N / pi)};
        }
        if (q > 1 && q < N && (N % q.get_si()) == 0) {
            int64_t qi = q.get_si();
            return {std::min(qi, N / qi), std::max(qi, N / qi)};
        }
        // trivial dependency (X = +-Y mod N) -- try the next one.
    }

    return {1, N};  // N is prime, or didn't converge with this many sr-pairs
}
