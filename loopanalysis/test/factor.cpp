#include <iostream>
using namespace std;

// Brute-force search over pairs (a, b) for a factor pair of n. This is the
// classical shape c2q_factor's Grover oracle actually accelerates -- an
// unstructured search over the full n x n space, not trial division up to
// sqrt(n). See analysis/factor/factor.md.
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
  int a, b;
  bool found = bruteForceFactor(91, a, b);
  cout << a << " " << b << " " << found << endl;
  return 0;
}
