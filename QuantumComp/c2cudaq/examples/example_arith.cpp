#include <c2cudaq.h>
#include <iostream>

int main() {
  std::cout << "=== Quantum Arithmetic Examples ===\n\n";

  // ADD
  // Default: bit width auto-computed from inputs
  std::cout << "-- c2q_add --\n";
  std::cout << "3 + 5 = " << c2q_add(3, 5) << "\n";
  std::cout << "12 + 15 = " << c2q_add(12, 15) << "\n";
  std::cout << "100 + 55 = " << c2q_add(100, 55) << "\n";

  // SUB
  std::cout << "\n-- c2q_sub --\n";
  std::cout << "8 - 3 = " << c2q_sub(8, 3) << "\n";
  std::cout << "3 - 8 = " << c2q_sub(3, 8) << "\n";  // negative result
  std::cout << "15 - 15 = " << c2q_sub(15, 15) << "\n";

  // MUL - auto bit width
  std::cout << "\n-- c2q_mul (auto bit width) --\n";
  std::cout << "3 * 5 = " << c2q_mul(3, 5) << "\n";
  std::cout << "7 * 11 = " << c2q_mul(7, 11) << "\n";

  // MUL - explicit bit widths (optimized for asymmetric inputs)
  std::cout << "\n-- c2q_mul (explicit size_a, size_b) --\n";
  // 13 needs 4 bits, 11 needs 4 bits
  std::cout << "13 * 11 = " << c2q_mul(13, 11, 4, 4) << "\n";
  // 2 needs 2 bits, 8 needs 4 bits
  std::cout << "2 * 8 = " << c2q_mul(2, 8, 2, 4) << "\n";

  return 0;
}
