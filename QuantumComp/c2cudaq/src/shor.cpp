#include <c2cudaq.h>
#include <cudaq.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Classical helpers

static int64_t shor_modpow(int64_t base, int64_t exp, int64_t mod) {
    int64_t r = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) r = r * base % mod;
        base = base * base % mod;
        exp >>= 1;
    }
    return r;
}

static int64_t shor_gcd(int64_t a, int64_t b) {
    while (b) { a %= b; std::swap(a, b); }
    return std::abs(a);
}

// Best rational approximation p/q of num/den with q ≤ max_q.
// Returns the denominator q of the best convergent.
static int64_t best_convergent_denom(int64_t num, int64_t den, int64_t max_q) {
    int64_t p0 = 0, p1 = 1, q0 = 1, q1 = 0;
    while (den > 0) {
        int64_t a  = num / den;
        int64_t p2 = a * p1 + p0;
        int64_t q2 = a * q1 + q0;
        if (q2 > max_q) break;
        p0 = p1; p1 = p2;
        q0 = q1; q1 = q2;
        int64_t rem = num - a * den;
        num = den; den = rem;
    }
    return q1;
}

// Builder API circuit primitives
// All functions below add gates to the kernel builder at C++ compile time.
// They are classical C++ functions that emit MLIR gate operations into the
// kernel; the resulting circuit runs on the QPU.

using Kernel = cudaq::kernel_builder<>;

// Mark anc ^= (ctrl=1 AND state == val):
//   X-wrap all 0-bits of val → all-ones MCX detection → X-wrap restore.
//   Applied unconditionally to the circuit; coherence is maintained because
//   the X-wrapping is symmetric (gate applied + same gate un-applied), so it
//   acts as identity on all states other than |val⟩ for the purpose of marking.
static void detect_state(Kernel& k, cudaq::QuakeValue ctrl,
                         cudaq::QuakeValue state, cudaq::QuakeValue anc,
                         int n, int val) {
    // Pre-flip 0-bits of val so |val⟩ → |1...1⟩
    for (int j = 0; j < n; ++j) {
        if (!((val >> j) & 1)) {
            auto qj = state[j];
            k.x(qj);
        }
    }
    // (n+1)-qubit MCX: ctrl + all state bits → anc
    std::vector<cudaq::QuakeValue> ctrls;
    ctrls.push_back(ctrl);
    for (int j = 0; j < n; ++j) ctrls.push_back(state[j]);
    k.x<cudaq::ctrl>(ctrls, anc);
    // Restore
    for (int j = 0; j < n; ++j) {
        if (!((val >> j) & 1)) {
            auto qj = state[j];
            k.x(qj);
        }
    }
}

// Controlled transposition |x⟩ ↔ |y⟩, controlled on ctrl.
// anc1, anc2 are scratch qubits that start and end in |0⟩.
//
// Protocol (all coherent):
//   1. anc1 = (ctrl=1 AND state=x)   [detect_state for x]
//   2. anc2 = (ctrl=1 AND state=y)
//   3. Flip all diff-bits of state conditioned on (anc1 OR anc2):
//        cx(anc1, state[b]) then cx(anc2, state[b]) for each b in diff(x,y)
//      → |x⟩ becomes |y⟩ (with anc1=1) and |y⟩ becomes |x⟩ (with anc2=1)
//   4. Uncompute: detect |y⟩ to reset anc1, detect |x⟩ to reset anc2.
static void ctrl_transposition(Kernel& k, cudaq::QuakeValue ctrl,
                               cudaq::QuakeValue state,
                               cudaq::QuakeValue anc1, cudaq::QuakeValue anc2,
                               int n, int x_val, int y_val) {
    if (x_val == y_val) return;

    detect_state(k, ctrl, state, anc1, n, x_val);
    detect_state(k, ctrl, state, anc2, n, y_val);

    int diff = x_val ^ y_val;
    for (int b = 0; b < n; ++b) {
        if ((diff >> b) & 1) {
            auto sb = state[b];
            k.x<cudaq::ctrl>(anc1, sb);
            auto sb2 = state[b];
            k.x<cudaq::ctrl>(anc2, sb2);
        }
    }

    // Uncompute: after the flips, former |x⟩ is now |y⟩ (anc1=1) and vice versa
    detect_state(k, ctrl, state, anc1, n, y_val);
    detect_state(k, ctrl, state, anc2, n, x_val);
}

// Controlled modular multiplication: state |x⟩ → |m·x mod N⟩, ctrl on ctrl.
// Decomposes the permutation x↦mx mod N into cycles, each implemented as a
// sequence of controlled transpositions using two shared ancilla qubits.
static void ctrl_mul_mod_N(Kernel& k, cudaq::QuakeValue ctrl,
                           cudaq::QuakeValue state,
                           cudaq::QuakeValue anc1, cudaq::QuakeValue anc2,
                           int n, int64_t m, int64_t N) {
    std::vector<int> perm(N);
    for (int i = 0; i < N; ++i) perm[i] = (int)((m * i) % N);

    std::vector<bool> visited(N, false);
    for (int start = 0; start < N; ++start) {
        if (visited[start] || perm[start] == start) {
            visited[start] = true;
            continue;
        }
        // Collect cycle starting at `start`
        std::vector<int> cycle;
        for (int cur = start; !visited[cur]; cur = perm[cur]) {
            visited[cur] = true;
            cycle.push_back(cur);
        }
        // Implement cycle (a₀→a₁→…→aₖ→a₀) as transpositions (a₀,a₁),(a₀,a₂),…
        for (size_t i = 1; i < cycle.size(); ++i)
            ctrl_transposition(k, ctrl, state, anc1, anc2, n,
                               cycle[0], cycle[i]);
    }
}

// IQFT on the first `t` qubits of `reg` (builder API version).
// Same convention as factor_qft_inv in grover.cpp: processes i=0..t-1 (LSB first).
static void apply_iqft_builder(Kernel& k, cudaq::QuakeValue reg, int t) {
    for (int i = 0; i < t; ++i) {
        for (int j = 0; j < i; ++j) {
            double angle = -M_PI / (double)(1LL << (i - j));
            std::vector<cudaq::QuakeValue> cv = {reg[j]};
            auto qi = reg[i];
            k.r1<cudaq::ctrl>(angle, cv, qi);
        }
        auto qi = reg[i];
        k.h(qi);
    }
}

// QPE circuit (builder API)
// Qubit layout:
//   counting[0..t-1]   : QPE counting register (H-initialised)
//   state[0..n-1]      : order-finding state register (init to |1⟩)
//   anc[0..1]          : ancilla for controlled permutation
//
// For n = ceil(log2(N)) and t = 2n:
//   Total qubits: t + n + 2 = 3n + 2 ≤ 28 for n ≤ 8 (N ≤ 255).
static cudaq::sample_result run_shor_qpe(int64_t N, int64_t a,
                                         int n_state, int t) {
    auto kernel  = cudaq::make_kernel();
    auto counting = kernel.qalloc(t);
    auto state    = kernel.qalloc(n_state);
    auto anc      = kernel.qalloc(2);

    // State register → |1⟩
    auto s0 = state[0];
    kernel.x(s0);

    // H on all counting qubits
    for (int j = 0; j < t; ++j) {
        auto cj = counting[j];
        kernel.h(cj);
    }

    // Controlled-U^(2^j) for j = 0..t-1
    // U|x⟩ = |a·x mod N⟩ → U^(2^j)|x⟩ = |a^(2^j)·x mod N⟩
    for (int j = 0; j < t; ++j) {
        int64_t power = shor_modpow(a, 1LL << j, N);
        auto cj = counting[j];
        auto anc0 = anc[0];
        auto anc1 = anc[1];
        ctrl_mul_mod_N(kernel, cj, state, anc0, anc1, n_state, power, N);
    }

    // IQFT on counting register
    apply_iqft_builder(kernel, counting, t);

    // Measure counting register
    kernel.mz(counting);

    return cudaq::sample(kernel);
}

// Public API
std::pair<int64_t, int64_t> c2q_factor_shor(int64_t n) {
    if (n < 4)
        throw std::invalid_argument("c2q_factor_shor: n must be >= 4");
    if (n % 2 == 0) return {2, n / 2};

    // n_state = number of bits needed to represent n
    int n_state = 0;
    { int64_t tmp = n; while (tmp) { ++n_state; tmp >>= 1; } }

    int t       = 2 * n_state;   // QPE counting bits (phase precision 2^t > N²)
    int total_q = t + n_state + 2;

    if (total_q > 28)
        throw std::runtime_error(
            "c2q_factor_shor: n requires " + std::to_string(total_q) +
            " qubits (simulator limit 28); safe range n ≤ 255");

    // Try bases a = 2, 3, ... until we find a factor
    for (int64_t a = 2; a < n; ++a) {
        // Lucky classical shortcut
        int64_t g = shor_gcd(a, n);
        if (g > 1) return {g, n / g};

        // Run QPE circuit (builder API)
        auto counts = run_shor_qpe(n, a, n_state, t);
        std::string bits = counts.most_probable();

        // Decode t-bit counting register: LSB-first → integer s
        std::string cb = bits.substr(0, (size_t)t);
        std::reverse(cb.begin(), cb.end());   // → MSB-first
        int64_t s = 0;
        for (char c : cb) s = (s << 1) | (c - '0');

        if (s == 0) continue;  // degenerate

        // Phase ≈ s / 2^t = k/r for some k coprime to r.
        // Find the best convergent denominator r ≤ n.
        int64_t two_t = 1LL << t;
        int64_t r     = best_convergent_denom(s, two_t, n);

        // Try r and 2r (common for even-order issue)
        for (int64_t cand_r : {r, 2 * r}) {
            if (cand_r < 2 || cand_r > n) continue;
            if (shor_modpow(a, cand_r, n) != 1) continue;   // not the order
            if (cand_r % 2 != 0) continue;                   // need even order

            int64_t sq = shor_modpow(a, cand_r / 2, n);
            for (int64_t delta : {-1LL, 1LL}) {
                int64_t factor = shor_gcd(sq + delta, n);
                if (factor > 1 && factor < n)
                    return {std::min(factor, n / factor),
                            std::max(factor, n / factor)};
            }
        }
    }

    return {1, n};   // prime or didn't converge
}
