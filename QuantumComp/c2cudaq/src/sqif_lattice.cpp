#include <c2cudaq/sqif.h>
#include <algorithm>
#include <cmath>
#include <random>

namespace c2cudaq {

// ---- small helpers ---------------------------------------------------------

static std::vector<int> sieve_primes_upto(int bound) {
    std::vector<int> primes;
    if (bound < 2) return primes;
    std::vector<bool> composite(bound + 1, false);
    for (int i = 2; i <= bound; ++i) {
        if (composite[i]) continue;
        primes.push_back(i);
        for (long long j = (long long)i * i; j <= bound; j += i) composite[j] = true;
    }
    return primes;
}

std::vector<int> sqif_first_n_primes(int n) {
    std::vector<int> primes;
    int candidate = 2;
    while ((int)primes.size() < n) {
        bool prime = true;
        for (int p : primes) {
            if ((long long)p * p > candidate) break;
            if (candidate % p == 0) { prime = false; break; }
        }
        if (prime) primes.push_back(candidate);
        ++candidate;
    }
    return primes;
}

std::vector<int> sqif_primes_upto(int B2) { return sieve_primes_upto(B2); }

std::vector<long> sqif_default_diag(int n) {
    std::vector<long> diag(n);
    for (int i = 0; i < n; ++i) diag[i] = (i + 2) / 2;  // ceil((i+1)/2)
    return diag;
}

std::vector<long> sqif_random_diag(int n, unsigned seed) {
    auto diag = sqif_default_diag(n);
    std::mt19937 rng(seed);
    std::shuffle(diag.begin(), diag.end(), rng);
    return diag;
}

// round to nearest integer, ties away from zero.
static mpz_class round_nearest(const mpq_class& q) {
    mpz_class num = q.get_num();
    mpz_class den = q.get_den();  // always > 0 for a canonicalized mpq_class
    int sign = (num >= 0) ? 1 : -1;
    mpz_class abs_two_num = 2 * abs(num);
    mpz_class val = (abs_two_num + den) / (2 * den);  // truncating div on non-negatives == floor
    return sign * val;
}

static mpz_class exact_mpz(const mpq_class& q) {
    mpq_class c = q;
    c.canonicalize();
    return c.get_num();  // caller guarantees denominator == 1
}

static mpq_class dot(const std::vector<mpq_class>& a, const std::vector<mpq_class>& b) {
    mpq_class s(0);
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

struct GramSchmidt {
    std::vector<std::vector<mpq_class>> bstar;  // n vectors, dim = n+1
    std::vector<std::vector<mpq_class>> mu;     // mu[i][j], j < i
    std::vector<mpq_class> normsq;
};

static GramSchmidt gram_schmidt(const SqifBasis& B) {
    int n = (int)B.size();
    int dim = (int)B[0].size();
    GramSchmidt gs;
    gs.bstar.assign(n, std::vector<mpq_class>(dim));
    gs.mu.assign(n, std::vector<mpq_class>(n, mpq_class(0)));
    gs.normsq.assign(n, mpq_class(0));
    for (int i = 0; i < n; ++i) {
        std::vector<mpq_class> bi(dim);
        for (int d = 0; d < dim; ++d) bi[d] = mpq_class(B[i][d]);
        std::vector<mpq_class> v = bi;
        for (int j = 0; j < i; ++j) {
            mpq_class m = dot(bi, gs.bstar[j]) / gs.normsq[j];
            gs.mu[i][j] = m;
            for (int d = 0; d < dim; ++d) v[d] -= m * gs.bstar[j][d];
        }
        gs.bstar[i] = v;
        gs.normsq[i] = dot(v, v);
    }
    return gs;
}

// ---- Stage 1: lattice construction -----------------------------------------

SqifLattice sqif_build_lattice(int64_t N, int n, double c, const std::vector<long>& diag) {
    SqifLattice L;
    L.n = n;
    L.primes = sqif_first_n_primes(n);
    double scale = std::pow(10.0, c);

    // Paper's "⌈x⌋" is round-to-nearest, not ceiling (verified against the
    // Eq. S35 worked values: 10^4*ln(2)=6931.47 rounds to 6931, but ceils
    // to 6932 -- the paper shows 6931).
    L.B.assign(n, std::vector<mpz_class>(n + 1, mpz_class(0)));
    for (int i = 0; i < n; ++i) {
        L.B[i][i] = mpz_class(diag[i]);
        long w = std::lround(scale * std::log((double)L.primes[i]));
        L.B[i][n] = mpz_class(w);
    }
    L.t.assign(n + 1, mpz_class(0));
    L.t[n] = mpz_class(std::lround(scale * std::log((double)N)));
    return L;
}

// ---- Stage 1: LLL reduction -------------------------------------------------

SqifBasis sqif_lll_reduce(const SqifBasis& Bin, double delta) {
    SqifBasis B = Bin;
    int n = (int)B.size();
    int dim = (int)B[0].size();
    mpq_class deltaq(delta);
    deltaq.canonicalize();

    GramSchmidt gs = gram_schmidt(B);
    int k = 1;
    while (k < n) {
        for (int j = k - 1; j >= 0; --j) {
            mpq_class mu = gs.mu[k][j];
            mpq_class half(1, 2);
            if (abs(mu) > half) {
                mpz_class r = round_nearest(mu);
                for (int d = 0; d < dim; ++d) B[k][d] -= r * B[j][d];
                gs = gram_schmidt(B);
            }
        }
        mpq_class lhs = gs.normsq[k];
        mpq_class rhs = (deltaq - gs.mu[k][k - 1] * gs.mu[k][k - 1]) * gs.normsq[k - 1];
        if (lhs >= rhs) {
            ++k;
        } else {
            std::swap(B[k], B[k - 1]);
            gs = gram_schmidt(B);
            k = std::max(k - 1, 1);
        }
    }
    return B;
}

// ---- Stage 1: Babai's nearest-plane algorithm -------------------------------

BabaiResult sqif_babai(const SqifBasis& D, const std::vector<mpz_class>& t) {
    int n = (int)D.size();
    int dim = (int)t.size();
    GramSchmidt gs = gram_schmidt(D);

    std::vector<mpq_class> b(dim);
    for (int d = 0; d < dim; ++d) b[d] = mpq_class(t[d]);

    BabaiResult res;
    res.steps.assign(n, BabaiStep{});
    for (int j = n - 1; j >= 0; --j) {
        mpq_class mu = dot(b, gs.bstar[j]) / gs.normsq[j];
        mpz_class c = round_nearest(mu);
        for (int d = 0; d < dim; ++d) b[d] -= mpq_class(c) * mpq_class(D[j][d]);
        res.steps[j] = BabaiStep{mu, c};
    }

    res.b_op.assign(dim, mpz_class(0));
    for (int d = 0; d < dim; ++d) {
        mpq_class diff = mpq_class(t[d]) - b[d];
        res.b_op[d] = exact_mpz(diff);
    }
    return res;
}

std::vector<int> sqif_encoding_signs(const BabaiResult& babai) {
    std::vector<int> signs(babai.steps.size());
    for (size_t i = 0; i < babai.steps.size(); ++i) {
        const auto& s = babai.steps[i];
        signs[i] = (s.c <= s.mu) ? +1 : -1;
    }
    return signs;
}

}  // namespace c2cudaq
