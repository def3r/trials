#include <c2cudaq.h>
#include <iostream>

int main() {
  std::cout << "=== Quantum Factorization Examples ===\n\n";

  // c2q_factor
  // Uses the QFT multiplier to quantum-verify each candidate factor pair.
  // Iterates classically but validates each pair on the quantum circuit.
  for (int64_t n : {15, 21, 35, 12, /*121*/}) {  // wat?? even 12 works!
    auto [p, q] = c2q_factor(n);
    std::cout << "c2q_factor(" << n << ") = {" << p << ", " << q << "}"
              << "   " << p << " * " << q << " = " << p * q
              << (p * q == n && p > 1 && q > 1 ? "  [valid]"
                                               : "  [trivial/error]")
              << "\n";
  }

  std::cout << "\n";

  // c2q_factor_sqif
  // Sublinear Quantum Integer Factorization (Yan et al., arXiv:2212.12372):
  // classical LLL + Babai sets up a closest-vector problem, QAOA refines
  // it, classical postprocessing (smoothness + GF(2)) extracts the
  // factors. Independent path from c2q_factor above -- not a replacement,
  // qubit count is sublinear in bit-length so it's meant to reach past
  // c2q_factor's ~128 ceiling. Currently validated against the paper's own
  // N=1961 worked example; see claude.md for status on larger N.
  for (int64_t n : {1961}) {
    auto [p, q] = c2q_factor_sqif(n);
    std::cout << "c2q_factor_sqif(" << n << ") = {" << p << ", " << q << "}"
              << "   " << p << " * " << q << " = " << p * q
              << (p * q == n && p > 1 && q > 1 ? "  [valid]"
                                               : "  [trivial/error]")
              << "\n";
  }

  return 0;
}
