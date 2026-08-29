#include <c2cudaq.h>
#include <cudaq.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// QFT helpers (same as libqmul)
struct qft_fwd {
  void operator()(cudaq::qview<> q) __qpu__ {
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

// Inverse QFT - no cudaq::adjoint (MLIR cloneReversedLoop crash on variable
// bounds)
struct qft_inv {
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

// Ripple-carry ADD Register layout:
// reg_a[0..n-1], reg_b[0..n-1], carry[0..n] (n+1 qubits). Result lands in
// carry[0..n].  carry[n] is the overflow/sign bit.
struct add_kernel {
  __qpu__ void operator()(int a, int b, int n, bool complement) {
    cudaq::qvector reg_a(n);
    cudaq::qvector reg_b(n);
    cudaq::qvector carry(n + 1);

    for (int i = 0; i < n; ++i) {
      if ((a >> i) & 1)
        x(reg_a[i]);

      if (complement) {
        if (!((b >> i) & 1))
          x(reg_b[i]);
      } else if ((b >> i) & 1)
        x(reg_b[i]);
    }
    if (complement)
      x(carry[0]);  // carry-in = 1 (two's complement)

    for (int i = 0; i < n; ++i) {
      x<cudaq::ctrl>(reg_a[i], reg_b[i], carry[i + 1]);  // CCX carry-out
      x<cudaq::ctrl>(reg_a[i], reg_b[i]);                // CX sum
      x<cudaq::ctrl>(reg_b[i], carry[i], carry[i + 1]);  // CCX propagate
      x<cudaq::ctrl>(reg_b[i], carry[i]);                // CX
      x<cudaq::ctrl>(reg_a[i], reg_b[i]);                // CX restore
    }
    mz(carry);
  }
};

// QFT multiplier
struct mul_kernel {
  __qpu__ void operator()(int a, int b, int sa, int sb) {
    cudaq::qvector ra(sa), rb(sb);
    int acc_size = sa + sb;
    cudaq::qvector acc(acc_size);

    for (int i = 0; i < sa; ++i)
      if ((a >> i) & 1)
        x(ra[i]);
    for (int j = 0; j < sb; ++j)
      if ((b >> j) & 1)
        x(rb[j]);

    qft_fwd{}(acc);

    for (int i = 0; i < sa; ++i)
      for (int j = 0; j < sb; ++j) {
        int p = i + j;
        for (int k = p; k < acc_size; ++k) {
          double angle = M_PI / (double)(1LL << (k - p));
          r1<cudaq::ctrl>(angle, ra[i], rb[j], acc[k]);
        }
      }

    qft_inv{}(acc);
    mz(acc);
  }
};

// Decode helpers
static int64_t bits_to_int(const std::string& bits) {
  std::string rev = bits;
  std::reverse(rev.begin(), rev.end());
  return (int64_t)std::stoull(rev, nullptr, 2);
}

// carry[n] = 1 → positive; = 0 → negative (two's complement, n bits)
static int64_t decode_carry(const std::string& bits, int n) {
  bool carry_out = (bits[n] == '1');
  std::string mag(bits.begin(), bits.begin() + n);
  std::reverse(mag.begin(), mag.end());
  int64_t v = (int64_t)std::stoull(mag, nullptr, 2);
  return carry_out ? v : v - (1LL << n);
}

// Public API
static int bit_length(int64_t x) {
  if (x <= 0)
    x = -x;
  if (x == 0)
    return 1;
  int b = 0;
  while (x) {
    ++b;
    x >>= 1;
  }
  return b;
}

int64_t c2q_add(int64_t a, int64_t b) {
  int n = std::max({bit_length(a), bit_length(b), 1}) + 1;
  if (3 * n + 1 > 28)
    throw std::runtime_error("c2q_add: inputs too large (limit |a|,|b| ≤ 255)");
  auto r = cudaq::sample(add_kernel{}, (int)a, (int)b, n, false);
  return bits_to_int(r.most_probable());
}

int64_t c2q_sub(int64_t a, int64_t b) {
  int n = std::max({bit_length(a), bit_length(b), 1}) + 1;
  if (3 * n + 1 > 28)
    throw std::runtime_error("c2q_sub: inputs too large (limit |a|,|b| ≤ 255)");
  auto r = cudaq::sample(add_kernel{}, (int)a, (int)b, n, true);
  return decode_carry(r.most_probable(), n);
}

int64_t c2q_mul(int64_t a, int64_t b) {
  int sa = bit_length(a), sb = bit_length(b);
  return c2q_mul(a, b, sa, sb);
}

int64_t c2q_mul(int64_t a, int64_t b, int size_a, int size_b) {
  if (size_a + size_b > 14)
    throw std::runtime_error(
        "c2q_mul: size_a + size_b > 14 exceeds qubit limit");
  auto r = cudaq::sample(mul_kernel{}, (int)a, (int)b, size_a, size_b);
  return bits_to_int(r.most_probable());
}
