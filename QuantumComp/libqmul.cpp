#include "libqmul.h"
#include <cudaq.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

struct apply_qft {
  void operator()(cudaq::qview<> qubits) __qpu__ {
    int M = qubits.size();
    for (int step_i = 0; step_i < M; ++step_i) {
      int i = M - 1 - step_i;
      h(qubits[i]);
      for (int step_j = 0; step_j < i; ++step_j) {
        int j = i - 1 - step_j;
        double angle = M_PI / (double)(1 << (i - j));
        r1<cudaq::ctrl>(angle, qubits[j], qubits[i]);
      }
    }
  }
};

// Inverse QFT implemented directly — avoids cudaq::adjoint whose MLIR
// cloneReversedLoop pass crashes on variable-bound inner loops.
struct apply_iqft {
  __qpu__ void operator()(cudaq::qview<> qubits) {
    int M = qubits.size();
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < i; ++j) {
        double angle = -M_PI / (double)(1 << (i - j));
        r1<cudaq::ctrl>(angle, qubits[j], qubits[i]);
      }
      h(qubits[i]);
    }
  }
};

struct qft_multiply_kernel {
  __qpu__ void operator()(int a, int b, int n_qubits) {
    cudaq::qvector reg_a(n_qubits);
    cudaq::qvector reg_b(n_qubits);
    cudaq::qvector acc(2 * n_qubits);

    for (int i = 0; i < n_qubits; ++i)
      if ((a >> i) & 1)
        x(reg_a[i]);
    for (int j = 0; j < n_qubits; ++j)
      if ((b >> j) & 1)
        x(reg_b[j]);

    apply_qft{}(acc);

    int M = 2 * n_qubits;
    for (int i = 0; i < n_qubits; ++i)
      for (int j = 0; j < n_qubits; ++j) {
        int p = i + j;
        for (int k = p; k < M; ++k) {
          double angle = M_PI / (double)(1 << (k - p));
          r1<cudaq::ctrl>(angle, reg_a[i], reg_b[j], acc[k]);
        }
      }

    apply_iqft{}(acc);
    mz(acc);
  }
};

struct qft_multiply_optimized_kernel {
  __qpu__ void operator()(int a, int b, int size_a, int size_b) {
    cudaq::qvector reg_a(size_a);
    cudaq::qvector reg_b(size_b);
    int acc_size = size_a + size_b;
    cudaq::qvector acc(acc_size);

    for (int i = 0; i < size_a; ++i)
      if ((a >> i) & 1)
        x(reg_a[i]);
    for (int j = 0; j < size_b; ++j)
      if ((b >> j) & 1)
        x(reg_b[j]);

    apply_qft{}(acc);

    for (int i = 0; i < size_a; ++i)
      for (int j = 0; j < size_b; ++j) {
        int p = i + j;
        for (int k = p; k < acc_size; ++k) {
          double angle = M_PI / (double)(1 << (k - p));
          r1<cudaq::ctrl>(angle, reg_a[i], reg_b[j], acc[k]);
        }
      }

    apply_iqft{}(acc);
    mz(acc);
  }
};

void print_as_decimal(const cudaq::sample_result& result,
                      const std::string& label) {
  std::cout << "--- Output for " << label << " ---\n";
  for (auto&& [bitstring, count] : result) {
    std::string rev = bitstring;
    std::reverse(rev.begin(), rev.end());
    unsigned long long val = std::stoull(rev, nullptr, 2);
    std::cout << "Raw: " << bitstring << " | Decimal: " << val
              << " | Shots: " << count << "\n";
  }
  std::cout << "\n";
}

cudaq::sample_result qft_multiply(int a, int b, int n_qubits) {
  return cudaq::sample(qft_multiply_kernel{}, a, b, n_qubits);
}

cudaq::sample_result qft_multiply_optimized(int a,
                                            int b,
                                            int size_a,
                                            int size_b) {
  return cudaq::sample(qft_multiply_optimized_kernel{}, a, b, size_a, size_b);
}
