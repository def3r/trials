#include <c2cudaq.h>
#include <chrono>
#include <iostream>
using namespace std;

static void tryN(long long n, long long expectA, long long expectB) {
  cout << "starting n=" << n << " ..." << endl;
  auto t0 = chrono::steady_clock::now();
  auto [a, b] = c2q_factor(n);
  auto t1 = chrono::steady_clock::now();
  double secs = chrono::duration<double>(t1 - t0).count();
  bool ok = (a * b == n) && ((a == expectA && b == expectB) || (a == expectB && b == expectA) || (a > 1 && b > 1));
  cout << "n=" << n << "  a=" << a << " b=" << b
       << "  " << (ok ? "OK" : "CHECK")
       << "  time=" << secs << "s" << endl;
}

int main() {
  // Single isolated tier-5 (64-127, 25 qubit) spot checks.
  tryN(85, 5, 17);   // 85 = 5*17
  tryN(91, 7, 13);   // 91 = 7*13
  return 0;
}
