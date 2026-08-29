#include <c2cudaq.h>
#include <cstdlib>
#include <iostream>

static int failures = 0;

static void check_factor(int64_t n, int64_t p, int64_t q) {
  // Accept any valid non-trivial factor pair in any order
  bool trivial = (p == 1 && q == n) || (p == n && q == 1);
  bool valid = (p * q == n) && !trivial;
  if (valid) {
    std::cout << "[PASS] c2q_factor(" << n << ") = {" << p << ", " << q
              << "}\n";
  } else if (trivial) {
    std::cout << "[FAIL] c2q_factor(" << n << "): returned trivial {" << p
              << ", " << q << "} — no non-trivial factor found\n";
    ++failures;
  } else {
    std::cout << "[FAIL] c2q_factor(" << n << "): {" << p << ", " << q
              << "} does not multiply to " << n << "\n";
    ++failures;
  }
}

int main() {
  auto [p1, q1] = c2q_factor(15);
  check_factor(15, p1, q1);  // 3*5
  auto [p2, q2] = c2q_factor(21);
  check_factor(21, p2, q2);  // 3*7
  auto [p3, q3] = c2q_factor(35);
  check_factor(35, p3, q3);  // 5*7
  auto [p4, q4] = c2q_factor(77);
  check_factor(77, p4, q4);  // 7*11
  auto [p5, q5] = c2q_factor(49);
  check_factor(49, p5, q5);  // 7*7

  std::cout << "\n"
            << (failures == 0 ? "All factor tests PASSED."
                              : std::to_string(failures) + " test(s) FAILED.")
            << "\n";
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
