#include <string>
#include "libqmul.h"

int main() {
  auto r1 = qft_multiply_optimized(13, 11, 4, 4);
  print_as_decimal(r1, "13 * 11");

  auto r2 = qft_multiply(7, 11, 4);
  print_as_decimal(r2, "7 * 11");

  auto r3 = qft_multiply(3, 5, 3);
  print_as_decimal(r3, "3 * 5");

  int a, b;
  while (1) {
    std::cin >> a >> b;
    auto res = qft_multiply(a, b, 4);
    print_as_decimal(
        res, std::string(std::to_string(a) + " * " + std::to_string(b)));
  }

  return 0;
}
