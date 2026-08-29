#include <c2cudaq/sqif.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace c2cudaq {

std::optional<SrPair> sqif_try_sr_pair(const mpz_class& u, const mpz_class& v,
                                        const std::vector<int>& u_exponents,
                                        int64_t N,
                                        const std::vector<int>& big_primes) {
    mpz_class diff = u - v * mpz_class(N);
    if (diff == 0) return std::nullopt;

    std::vector<int> d_exponents(big_primes.size() + 1, 0);
    d_exponents[0] = (diff < 0) ? 1 : 0;

    mpz_class rem = abs(diff);
    for (size_t k = 0; k < big_primes.size(); ++k) {
        mpz_class p(big_primes[k]);
        int e = 0;
        mpz_class q, r;
        while (true) {
            mpz_fdiv_qr(q.get_mpz_t(), r.get_mpz_t(), rem.get_mpz_t(), p.get_mpz_t());
            if (r != 0) break;
            rem = q;
            ++e;
        }
        d_exponents[k + 1] = e;
        if (rem == 1) break;
    }
    if (rem != 1) return std::nullopt;  // not B2-smooth

    SrPair sr;
    sr.u = u;
    sr.v = v;
    sr.u_exponents = u_exponents;
    sr.d_exponents = std::move(d_exponents);
    return sr;
}

// ---- GF(2) Gaussian elimination --------------------------------------------
// Combined system: [u_exponents (small_dim cols) | d_exponents (big_dim
// cols)]. A dependent subset must zero out both halves simultaneously --
// see the note on SrPair in sqif.h for why this is two constraints, not
// one.

using Bits = std::vector<uint64_t>;

static void xor_into(Bits& a, const Bits& b) {
    for (size_t i = 0; i < a.size(); ++i) a[i] ^= b[i];
}

static int lowest_set_bit(const Bits& a) {
    for (size_t w = 0; w < a.size(); ++w) {
        if (a[w] == 0) continue;
        for (int b = 0; b < 64; ++b)
            if ((a[w] >> b) & 1ULL) return (int)(w * 64 + (size_t)b);
    }
    return -1;
}

std::vector<std::vector<int>> sqif_gf2_dependencies(const std::vector<SrPair>& pairs,
                                                      int small_dim, int big_dim) {
    int m = (int)pairs.size();
    if (m == 0) return {};
    int basis_dim = small_dim + big_dim;
    int wb = (basis_dim + 63) / 64;
    int wc = (m + 63) / 64;

    std::vector<Bits> rows(m, Bits(wb, 0));
    std::vector<Bits> comb(m, Bits(wc, 0));
    for (int i = 0; i < m; ++i) {
        int lim_u = std::min(small_dim, (int)pairs[i].u_exponents.size());
        for (int k = 0; k < lim_u; ++k)
            if (pairs[i].u_exponents[k] & 1) rows[i][k / 64] |= (1ULL << (k % 64));
        int lim_d = std::min(big_dim, (int)pairs[i].d_exponents.size());
        for (int k = 0; k < lim_d; ++k) {
            int col = small_dim + k;
            if (pairs[i].d_exponents[k] & 1) rows[i][col / 64] |= (1ULL << (col % 64));
        }
        comb[i][i / 64] |= (1ULL << (i % 64));
    }

    std::vector<int> pivot_of_col(basis_dim, -1);
    std::vector<std::vector<int>> deps;
    for (int i = 0; i < m; ++i) {
        Bits r = rows[i], c = comb[i];
        while (true) {
            int lo = lowest_set_bit(r);
            if (lo < 0) {
                std::vector<int> subset;
                for (int j = 0; j < m; ++j)
                    if ((c[j / 64] >> (j % 64)) & 1ULL) subset.push_back(j);
                if (!subset.empty()) deps.push_back(std::move(subset));
                break;
            }
            if (pivot_of_col[lo] < 0) {
                pivot_of_col[lo] = i;
                rows[i] = r;
                comb[i] = c;
                break;
            }
            xor_into(r, rows[pivot_of_col[lo]]);
            xor_into(c, comb[pivot_of_col[lo]]);
        }
    }
    return deps;
}

// ---- factor extraction ------------------------------------------------------
// X = sqrt(prod u_j), Y = sqrt(prod |u_j - v_j*N|) over the subset -- see
// sqif.h's note on SrPair for why X is a square root here, not prod u_j
// directly (verified against the paper's own "10th+17th sr-pair" worked
// example, Eq. S57/S61: X=5400=sqrt(u_10*u_17), Y=1001=sqrt(d_10*d_17)).

std::pair<mpz_class, mpz_class> sqif_extract_factors(const std::vector<SrPair>& pairs,
                                                       const std::vector<int>& subset,
                                                       int64_t N) {
    mpz_class Nz(N);
    mpz_class u_prod(1), d_prod(1);
    for (int idx : subset) {
        u_prod *= pairs[idx].u;
        d_prod *= (pairs[idx].u - pairs[idx].v * Nz);
    }

    auto isqrt_exact = [](const mpz_class& val, const char* what) {
        mpz_class abs_val = abs(val), root;
        mpz_sqrt(root.get_mpz_t(), abs_val.get_mpz_t());
        if (root * root != abs_val)
            throw std::runtime_error(
                std::string("sqif_extract_factors: ") + what +
                " is not a perfect square -- GF(2) dependency is inconsistent");
        return root;
    };

    mpz_class X = isqrt_exact(u_prod, "product of u_j over the subset");
    mpz_class Y = isqrt_exact(d_prod, "product of (u_j - v_j*N) over the subset");

    mpz_class sum = X + Y, diff = X - Y;
    mpz_class p, q;
    mpz_gcd(p.get_mpz_t(), sum.get_mpz_t(), Nz.get_mpz_t());
    mpz_gcd(q.get_mpz_t(), diff.get_mpz_t(), Nz.get_mpz_t());
    return {p, q};
}

}  // namespace c2cudaq
