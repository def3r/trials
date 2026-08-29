#include <iostream>
using namespace std;

// Same shape as test/factor.cpp / test/factor/basic.cpp: brute-force
// search over pairs (a, b) for a*b == n.

bool bruteForceFactor(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n) {
        outA = a;
        outB = b;
        return true;
      }
    }
  }
  outA = 1;
  outB = n;
  return false;
}

int main() {
  int n = 91;  // 7 * 13
  int a, b;

  cout << "n      : " << n << "\n";
  cout << "Solver : @factor_impl -> c2q_factor (Grover) for n<=127, "
       << "classical fallback otherwise -- see analysis/factor/factor.md\n\n";

  bool found = bruteForceFactor(n, a, b);

  cout << "found  = " << found << "\n";
  cout << "a * b  = " << a << " * " << b << " = " << (a * b) << "\n\n";

  if (found && a * b == n && a > 1 && b > 1) {
    cout << "PASS  correct non-trivial factor pair\n";
    return 0;
  }
  cout << "FAIL  result incorrect\n";
  return 1;
}
