#include <c2cudaq/sqif.h>
#include <algorithm>

namespace c2cudaq {

// Hc = ||t - sum_i x_i*d_i - b_op||^2, x_i substituted by the operator
// (s_i/2)(I - Z_i) (paper Eq. S48). Expanding (derivation in claude.md):
//   e[j]  = t[j] - b_op[j]
//   B[i][j] = (s_i/2) * D[i][j]
//   A[j]  = sum_i B[i][j];  e'[j] = e[j] - A[j]
//   offset = sum_j e'[j]^2 + sum_i sum_j B[i][j]^2
//   h_i   = 2 * sum_j e'[j] * B[i][j]
//   J_ik  = 2 * sum_j B[i][j] * B[k][j]           (i < k)
IsingTerms sqif_build_hamiltonian(const SqifBasis& D,
                                   const std::vector<mpz_class>& t,
                                   const std::vector<mpz_class>& b_op,
                                   const std::vector<int>& signs) {
    int n = (int)D.size();
    int dim = (int)t.size();

    std::vector<mpq_class> e(dim);
    for (int j = 0; j < dim; ++j) e[j] = mpq_class(t[j]) - mpq_class(b_op[j]);

    std::vector<std::vector<mpq_class>> B(n, std::vector<mpq_class>(dim));
    for (int i = 0; i < n; ++i) {
        mpq_class half_s(signs[i], 2);
        half_s.canonicalize();
        for (int j = 0; j < dim; ++j) B[i][j] = half_s * mpq_class(D[i][j]);
    }

    std::vector<mpq_class> ep(dim);
    for (int j = 0; j < dim; ++j) {
        mpq_class A(0);
        for (int i = 0; i < n; ++i) A += B[i][j];
        ep[j] = e[j] - A;
    }

    mpq_class offset(0);
    for (int j = 0; j < dim; ++j) offset += ep[j] * ep[j];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < dim; ++j) offset += B[i][j] * B[i][j];

    std::vector<mpq_class> h(n, mpq_class(0));
    for (int i = 0; i < n; ++i) {
        mpq_class s(0);
        for (int j = 0; j < dim; ++j) s += ep[j] * B[i][j];
        h[i] = 2 * s;
    }

    IsingTerms out;
    out.offset = offset.get_d();
    for (int i = 0; i < n; ++i) {
        if (h[i] != 0) {
            out.z_i.push_back(i);
            out.z_c.push_back(h[i].get_d());
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int kk = i + 1; kk < n; ++kk) {
            mpq_class s(0);
            for (int j = 0; j < dim; ++j) s += B[i][j] * B[kk][j];
            mpq_class J = 2 * s;
            if (J != 0) {
                out.zz_i.push_back(i);
                out.zz_j.push_back(kk);
                out.zz_c.push_back(J.get_d());
            }
        }
    }
    return out;
}

// bits[i] (qaoa_general's convention, matching cudaq::sample's bit order)
// -> x_i in {0, signs[i]}. Bit '1' means the qubit measured |1>, which
// under x_i = (s_i/2)(1 - z_i) with z_i = +1 for |0>, -1 for |1>,
// corresponds to x_i = s_i (bit '0' -> x_i = 0).
bool sqif_decode_uv(const std::string& bits, const SqifBasis& D,
                     const std::vector<mpz_class>& b_op,
                     const std::vector<int>& signs,
                     const std::vector<int>& small_primes,
                     mpz_class& u, mpz_class& v,
                     std::vector<int>& u_exponents) {
    int n = (int)D.size();
    if ((int)bits.size() < n) return false;

    std::vector<mpz_class> v_new = b_op;
    for (int i = 0; i < n; ++i) {
        if (bits[i] == '1') {
            int x = signs[i];
            for (size_t d = 0; d < v_new.size(); ++d)
                v_new[d] += mpz_class(x) * D[i][d];
        }
    }

    u = mpz_class(1);
    v = mpz_class(1);
    u_exponents.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        if (!v_new[i].fits_slong_p()) return false;
        long e = v_new[i].get_si();
        u_exponents[i] = (int)std::max(e, 0L);
        if (e > 0) {
            mpz_class pw;
            mpz_pow_ui(pw.get_mpz_t(), mpz_class(small_primes[i]).get_mpz_t(), (unsigned long)e);
            u *= pw;
        } else if (e < 0) {
            mpz_class pw;
            mpz_pow_ui(pw.get_mpz_t(), mpz_class(small_primes[i]).get_mpz_t(), (unsigned long)(-e));
            v *= pw;
        }
    }
    return true;
}

}  // namespace c2cudaq
