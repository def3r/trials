#include <c2cudaq.h>
#include <cudaq.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

// QFT forward
struct factor_qft_fwd {
    __qpu__ void operator()(cudaq::qview<> q) {
        int M = q.size();
        for (int si = 0; si < M; ++si) {
            int i = M - 1 - si;
            h(q[i]);
            for (int sj = 0; sj < i; ++sj) {
                int j = i - 1 - sj;
                double angle = M_PI / (double)(1 << (i - j));
                r1<cudaq::ctrl>(angle, q[j], q[i]);
            }
        }
    }
};

// QFT inverse
struct factor_qft_inv {
    __qpu__ void operator()(cudaq::qview<> q) {
        int M = q.size();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < i; ++j) {
                double angle = -M_PI / (double)(1 << (i - j));
                r1<cudaq::ctrl>(angle, q[j], q[i]);
            }
            h(q[i]);
        }
    }
};

// Grover factoring kernel
// Register layout (all allocated together so oracle and diffuser share qubits):
//   work[0..sa-1]     = register a  (|a⟩, searched in superposition)
//   work[sa..sa+sb-1] = register b  (|b⟩, searched in superposition)
//   acc[0..sa+sb-1]   = product accumulator (initialized and restored to |0⟩)
//   anc[0]            = phase ancilla (held in |−⟩ for kickback across all iters)
//
// Each Grover iteration:
//   1. Forward QFT-multiply:    acc ← a * b  (in superposition)
//   2. Phase oracle:            flip phase of |a,b⟩ states where a*b = n
//   3. Inverse QFT-multiply:    acc ← |0⟩   (uncompute, no cudaq::adjoint)
//   4. Grover diffuser:         amplify amplitude of solutions in |work⟩
struct factor_grover_kernel {
    __qpu__ void operator()(int n, int sa, int sb, int iters) {
        int nwork    = sa + sb;
        int acc_size = sa + sb;

        cudaq::qvector work(nwork);
        cudaq::qvector acc(acc_size);
        cudaq::qvector anc(1);

        // Uniform superposition over all candidate factor pairs |a⟩|b⟩
        for (int j = 0; j < nwork; ++j) h(work[j]);

        // Phase ancilla stays in |−⟩ = (|0⟩-|1⟩)/√2 throughout all iterations.
        // MCX into this ancilla produces phase kickback without disturbing it.
        x(anc[0]);
        h(anc[0]);

        for (int iter = 0; iter < iters; ++iter) {

            // 1. Forward QFT multiply: acc = work_a * work_b
            // work[0..sa-1] = a bits (qubit i = bit i of a)
            // work[sa..sa+sb-1] = b bits (qubit sa+j = bit j of b)
            factor_qft_fwd{}(acc);
            for (int i = 0; i < sa; ++i)
                for (int j = 0; j < sb; ++j) {
                    int p = i + j;
                    for (int k = p; k < acc_size; ++k) {
                        double angle = M_PI / (double)(1LL << (k - p));
                        r1<cudaq::ctrl>(angle, work[i], work[sa + j], acc[k]);
                    }
                }
            factor_qft_inv{}(acc);

            // 2. Phase oracle: flip phase if acc == n
            // Flip bits where n has 0 so that n maps to the all-ones pattern.
            for (int k = 0; k < acc_size; ++k)
                if (!((n >> k) & 1)) x(acc[k]);
            // MCX: all-ones in acc → kick phase into anc[0] via phase kickback.
            x<cudaq::ctrl>(acc, anc[0]);
            // Restore acc bit flips.
            for (int k = 0; k < acc_size; ++k)
                if (!((n >> k) & 1)) x(acc[k]);

            // 3. Inverse QFT multiply: restore acc to |0⟩
            // (QFT · neg-phases · IQFT) is the inverse of (QFT · phases · IQFT)
            // because (ABC)† = C† B† A† and R1(θ)† = R1(-θ), IQFT† = QFT.
            factor_qft_fwd{}(acc);
            for (int i = 0; i < sa; ++i)
                for (int j = 0; j < sb; ++j) {
                    int p = i + j;
                    for (int k = p; k < acc_size; ++k) {
                        double angle = -M_PI / (double)(1LL << (k - p));
                        r1<cudaq::ctrl>(angle, work[i], work[sa + j], acc[k]);
                    }
                }
            factor_qft_inv{}(acc);

            // 4. Grover diffuser on work
            // 2|s⟩⟨s| - I = H(2|0⟩⟨0| - I)H
            // 2|0⟩⟨0| - I = X⊗n · (H · MCX · H on last) · X⊗n
            for (int j = 0; j < nwork; ++j) h(work[j]);
            for (int j = 0; j < nwork; ++j) x(work[j]);
            h(work[nwork - 1]);
            auto work_ctrl = work.front(nwork - 1);
            x<cudaq::ctrl>(work_ctrl, work[nwork - 1]);
            h(work[nwork - 1]);
            for (int j = 0; j < nwork; ++j) x(work[j]);
            for (int j = 0; j < nwork; ++j) h(work[j]);
        }

        mz(work);
    }
};

// Decode: split work bitstring into (a, b)
// CUDA-Q most_probable() returns bits with qubit 0 first (LSB first).
// work = [a_bits | b_bits], so string[0..sa-1] = a, string[sa..sa+sb-1] = b.
static std::pair<int64_t, int64_t>
decode_factors(const std::string& bits, int sa, int sb) {
    auto half = [&](int start, int len) -> int64_t {
        std::string sub = bits.substr(start, len);
        std::reverse(sub.begin(), sub.end()); // make MSB-first for stoull
        return (int64_t)std::stoull(sub, nullptr, 2);
    };
    return {half(0, sa), half(sa, sb)};
}

// Public API
std::pair<int64_t, int64_t> c2q_factor(int64_t n) {
    if (n < 4)
        throw std::invalid_argument("c2q_factor: n must be >= 4");

    int num_result = 0;
    { int64_t tmp = n; while (tmp) { ++num_result; tmp >>= 1; } }

    // Use num_result-1 bits per factor register so both registers can represent
    // any factor of n (factors range from 2 to n/2, needing up to num_result-1 bits).
    // Using ceil(num_result/2) (the Python default) is wrong for asymmetric semiprimes
    // like 15=3*5: the small factor (3) fits in 2 bits but the large one (5) does not,
    // so the product register never reaches n and the oracle marks nothing.
    int num_state = num_result - 1;

    // Total: work(2*num_state) + acc(2*num_state) + anc(1)
    int total_q = 4 * num_state + 1;
    if (total_q > 28)
        throw std::runtime_error(
            "c2q_factor: n requires " + std::to_string(total_q) +
            " qubits (simulator limit 28); safe range is n <= 127 (num_result <= 7)");

    // Optimal Grover iterations ≈ (π/4) * sqrt(N/M)
    // N = 2^(2*num_state) candidates, M = 2 for semiprime (both (p,q) and (q,p))
    int iters = std::max(1, (int)std::round(
        (M_PI / 4.0) * std::pow(2.0, (double)num_state - 0.5)));

    auto counts = cudaq::sample(factor_grover_kernel{},
                                (int)n, num_state, num_state, iters);

    // Most probable state should be the amplified factor pair after Grover.
    auto [a, b] = decode_factors(counts.most_probable(), num_state, num_state);

    if (a > 1 && b > 1 && a * b == n)
        return {std::min(a, b), std::max(a, b)};

    return {1, n};  // n is prime or didn't converge this run
}
