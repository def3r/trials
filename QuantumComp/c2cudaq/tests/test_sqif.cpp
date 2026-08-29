// Regression tests for the SQIF factoring path (src/sqif_*.cpp), checked
// against arXiv:2212.12372's own worked examples wherever the paper gives
// exact numbers to check against. See claude.md for the equation mapping
// and for why LLL-reduced-basis matrices specifically are *not* asserted
// bit-for-bit (LLL reduction isn't unique).
#include <c2cudaq.h>
#include <c2cudaq/sqif.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace c2cudaq;

static int failures = 0;

static void expect(bool cond, const std::string& what) {
    if (cond) {
        std::cout << "[PASS] " << what << "\n";
    } else {
        std::cout << "[FAIL] " << what << "\n";
        ++failures;
    }
}

static void expect_eq(const mpz_class& got, const mpz_class& want, const std::string& what) {
    expect(got == want, what + " (got " + got.get_str() + ", want " + want.get_str() + ")");
}

// ---- Stage 1: lattice construction, exact match against Eq. S37/S35/S36 --

static void test_lattice_3qubit() {
    // N=1961, n=3, c=1.5, diag={1,1,2} -> paper's B_{3,1.5}/t_3 (Eq. S37).
    auto L = sqif_build_lattice(1961, 3, 1.5, {1, 1, 2});
    // Columns: (1,0,0,22), (0,1,0,35), (0,0,2,51); t=(0,0,0,240).
    expect_eq(L.B[0][0], 1, "3-qubit lattice col0[0]");
    expect_eq(L.B[0][3], 22, "3-qubit lattice col0 weight row");
    expect_eq(L.B[1][1], 1, "3-qubit lattice col1[1]");
    expect_eq(L.B[1][3], 35, "3-qubit lattice col1 weight row");
    expect_eq(L.B[2][2], 2, "3-qubit lattice col2[2]");
    expect_eq(L.B[2][3], 51, "3-qubit lattice col2 weight row");
    expect_eq(L.t[3], 240, "3-qubit target weight row");
}

static void test_lattice_5qubit() {
    // N=48567227, n=5, c=4, diag={2,1,3,2,1} -> paper's B_{5,4}/t_5
    // (Eq. S35/S36).
    auto L = sqif_build_lattice(48567227, 5, 4.0, {2, 1, 3, 2, 1});
    long expect_diag[5] = {2, 1, 3, 2, 1};
    long expect_weight[5] = {6931, 10986, 16094, 19459, 23979};
    for (int i = 0; i < 5; ++i) {
        expect_eq(L.B[i][i], expect_diag[i], "5-qubit lattice diag[" + std::to_string(i) + "]");
        expect_eq(L.B[i][5], expect_weight[i], "5-qubit lattice weight[" + std::to_string(i) + "]");
    }
    expect_eq(L.t[5], 176985, "5-qubit target weight row");
}

// ---- Stage 1/2: Babai + Hamiltonian, exact match against the paper's own
// LLL-reduced basis (Eq. S40/S41) -- bypasses our LLL entirely so this
// isolates correctness of Babai/Hamiltonian from LLL's non-uniqueness. ---

static SqifBasis basis_from_columns(std::initializer_list<std::initializer_list<long>> cols) {
    SqifBasis B;
    for (auto& col : cols) {
        std::vector<mpz_class> v;
        for (long x : col) v.push_back(mpz_class(x));
        B.push_back(v);
    }
    return B;
}

static void test_babai_3qubit() {
    // D_{3,1.5} (Eq. S40), columns of the printed matrix.
    auto D = basis_from_columns({{1, -2, 2, 3}, {-4, 1, 2, -2}, {-3, 2, 0, 4}});
    std::vector<mpz_class> t = {0, 0, 0, 240};
    auto babai = sqif_babai(D, t);
    // Expected b_op (Eq. S44): (0,4,4,242).
    mpz_class expect_bop[4] = {0, 4, 4, 242};
    for (int i = 0; i < 4; ++i)
        expect_eq(babai.b_op[i], expect_bop[i], "3-qubit b_op[" + std::to_string(i) + "]");

    auto signs = sqif_encoding_signs(babai);
    // Table S3: all three steps rounded up (c > mu) -> sign -1 for all.
    for (int i = 0; i < 3; ++i)
        expect(signs[i] == -1, "3-qubit sign[" + std::to_string(i) + "] == -1");

    auto ising = sqif_build_hamiltonian(D, t, babai.b_op, signs);
    // Eq. S51: Hc3 = 43.5I - 4*z1*z2 + 2.5*z1*z3 - 1.5*z1 + 3*z2*z3 - 3.5*z2 - 4*z3.
    expect(std::abs(ising.offset - 43.5) < 1e-9, "3-qubit Hamiltonian offset == 43.5");
    // z linear terms, indexed 0=x1,1=x2,2=x3.
    double want_h[3] = {-1.5, -3.5, -4.0};
    for (size_t k = 0; k < ising.z_i.size(); ++k)
        expect(std::abs(ising.z_c[k] - want_h[ising.z_i[k]]) < 1e-9,
               "3-qubit h[" + std::to_string(ising.z_i[k]) + "] matches Eq. S51");
    expect(ising.z_i.size() == 3, "3-qubit Hamiltonian has 3 linear terms");
    expect(ising.zz_i.size() == 3, "3-qubit Hamiltonian has 3 ZZ terms");
}

static void test_babai_5qubit() {
    // D_{5,4} (Eq. S41), columns of the printed matrix.
    auto D = basis_from_columns({
        {6, -4, 6, 4, -2, -3},
        {-8, -3, 6, -2, 2, 5},
        {2, 11, 3, 0, -6, -3},
        {-4, -5, 0, 12, -2, 4},
        {-4, -3, -3, 4, 1, -17},
    });
    std::vector<mpz_class> t = {0, 0, 0, 0, 0, 176985};
    auto babai = sqif_babai(D, t);
    // Expected b_op (Eq. S43): (2,4,9,8,0,176993).
    mpz_class expect_bop[6] = {2, 4, 9, 8, 0, 176993};
    for (int i = 0; i < 6; ++i)
        expect_eq(babai.b_op[i], expect_bop[i], "5-qubit b_op[" + std::to_string(i) + "]");

    auto signs = sqif_encoding_signs(babai);
    // Table S2: x1..x4 rounded up (-1), x5 rounded down (+1).
    int want_signs[5] = {-1, -1, -1, -1, +1};
    for (int i = 0; i < 5; ++i)
        expect(signs[i] == want_signs[i], "5-qubit sign[" + std::to_string(i) + "]");
}

// ---- Stage 3: smoothness test + GF(2) + factor extraction, using the
// paper's own published sr-pairs for N=1961 (Section VI.A) as a fixture --
// this validates postprocessing deterministically, without depending on
// QAOA actually finding these pairs itself. --------------------------------

struct KnownSrPair {
    long u, v;
};

static void test_smoothness_and_factor_extraction() {
    const int64_t N = 1961;
    auto big_primes = sqif_primes_upto(47);  // Table S5: B2=47 for 3-qubit.
    expect(big_primes.size() == 15, "3-qubit B2-bounded prime basis has 15 primes");

    // Paper's 20 published sr-pairs for N=1961 (page 22 of the supplement).
    // u given directly (already a product of the small basis {2,3,5}); v
    // as published. u_exponents recomputed from N's small basis {2,3,5}
    // via trial division here, purely for this fixture -- the real
    // pipeline gets it from sqif_decode_uv instead.
    std::vector<KnownSrPair> known = {
        {1944, 1}, {2000, 1}, {1920, 1}, {2025, 1}, {1875, 1},
        {1800, 1}, {2250, 1}, {1620, 1}, {1600, 1}, {2500, 1},
        {1350, 1}, {1296, 1}, {1125, 1}, {1000, 1}, {972, 1},
        {800, 1},  {11664, 5}, {3888, 1}, {6075, 1}, {9375, 2},
    };
    auto small_primes = sqif_first_n_primes(3);  // {2,3,5}

    std::vector<SrPair> pairs;
    for (auto& kp : known) {
        mpz_class u(kp.u), v(kp.v);
        std::vector<int> u_exp(3, 0);
        mpz_class rem = u;
        for (int i = 0; i < 3; ++i) {
            mpz_class p(small_primes[i]);
            while (rem % p == 0) { rem /= p; ++u_exp[i]; }
        }
        expect(rem == 1, "known sr-pair u=" + std::to_string(kp.u) +
                              " is smooth over {2,3,5}");
        if (auto sr = sqif_try_sr_pair(u, v, u_exp, N, big_primes))
            pairs.push_back(*sr);
        else
            expect(false, "known sr-pair u=" + std::to_string(kp.u) + " should be B2-smooth");
    }
    expect(pairs.size() == known.size(), "all 20 known sr-pairs accepted");

    auto deps = sqif_gf2_dependencies(pairs, 3, (int)big_primes.size() + 1);
    expect(!deps.empty(), "GF(2) elimination finds at least one dependency among the 20 pairs");

    // Paper's Eg.1 (page 22): the 4th pair alone (u=2025=3^4*5^2, v=1,
    // |u-vN|=2^6) is already a valid single-pair dependency (all exponents
    // even), giving X=sqrt(2025)=45, Y=sqrt(64)=8, p=gcd(53),q=gcd(37).
    bool found_single4 = false;
    for (auto& d : deps) {
        if (d.size() == 1 && d[0] == 3) {  // index 3 == the 4th pair (0-indexed)
            found_single4 = true;
            auto [p, q] = sqif_extract_factors(pairs, d, N);
            bool ok = (p == 53 && q == 37) || (p == 37 && q == 53);
            expect(ok, "Eg.1: 4th sr-pair alone factors 1961 as 37*53");
        }
    }
    expect(found_single4, "GF(2) elimination recovers the trivial single-pair (4th) dependency");

    // Paper's Eg.3 (Eq. S57/S61): combination of the 10th (idx 9) and 17th
    // (idx 16) pairs -- X=5400, Y=1001, p=53, q=37.
    std::vector<int> combo = {9, 16};
    auto [p3, q3] = sqif_extract_factors(pairs, combo, N);
    bool ok3 = (p3 == 53 && q3 == 37) || (p3 == 37 && q3 == 53);
    expect(ok3, "Eg.3: 10th+17th sr-pair combination factors 1961 as 37*53");
}

// ---- End-to-end: the real pipeline (QAOA sampling included) --------------

static void check_factor(int64_t n, int64_t p, int64_t q) {
    bool trivial = (p == 1 && q == n) || (p == n && q == 1);
    bool valid = (p * q == n) && !trivial;
    if (valid) {
        std::cout << "[PASS] c2q_factor_sqif(" << n << ") = {" << p << ", " << q << "}\n";
    } else if (trivial) {
        std::cout << "[FAIL] c2q_factor_sqif(" << n << "): returned trivial {" << p << ", "
                  << q << "}\n";
        ++failures;
    } else {
        std::cout << "[FAIL] c2q_factor_sqif(" << n << "): {" << p << ", " << q
                  << "} does not multiply to " << n << "\n";
        ++failures;
    }
}

int main() {
    test_lattice_3qubit();
    test_lattice_5qubit();
    test_babai_3qubit();
    test_babai_5qubit();
    test_smoothness_and_factor_extraction();

    // Real end-to-end pipeline, smallest validated case only (5/10-qubit
    // cases are structurally supported but not exercised here -- see
    // claude.md's test plan for why: larger B2/basis sizes make this a
    // slower, more QAOA-sampling-variance-dependent run).
    auto [p, q] = c2q_factor_sqif(1961);
    check_factor(1961, p, q);

    std::cout << "\n"
              << (failures == 0 ? "All SQIF tests PASSED."
                                : std::to_string(failures) + " test(s) FAILED.")
              << "\n";
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
