#include <c2cudaq.h>
#include <chrono>
#include <iostream>
using namespace std;

static bool isPrimeClassical(long long n) {
  if (n < 2) return false;
  for (long long d = 2; d * d <= n; d++)
    if (n % d == 0) return false;
  return true;
}

static void tryN(long long n) {
  auto t0 = chrono::steady_clock::now();
  auto [a, b] = c2q_factor(n);
  auto t1 = chrono::steady_clock::now();
  double secs = chrono::duration<double>(t1 - t0).count();
  bool prime = isPrimeClassical(n);
  bool ok = prime ? (a == 1 && b == n) : (a > 1 && b > 1 && a * b == n);
  cout << "n=" << n << (prime ? " (prime)" : " (composite)")
       << "  a=" << a << " b=" << b
       << "  " << (ok ? "OK" : "WRONG")
       << "  time=" << secs << "s" << endl;
}

int main() {
  cout << "=== Tier 1: n=4-7, 9 qubits ===" << endl;
  for (long long n : {4, 5, 6, 7}) tryN(n);

  cout << "\n=== Tier 2: n=8-15, 13 qubits ===" << endl;
  for (long long n : {8, 9, 11, 13, 14, 15}) tryN(n);

  cout << "\n=== Tier 3: n=16-31, 17 qubits ===" << endl;
  for (long long n : {16, 17, 21, 25, 29, 31}) tryN(n);

  cout << "\n=== Tier 4: n=32-63, 21 qubits ===" << endl;
  for (long long n : {32, 35, 47, 51, 61, 63}) tryN(n);

  cout << "\n=== Repeat n=15 x5 (consistency check, no seed param) ===" << endl;
  for (int i = 0; i < 5; i++) tryN(15);

  cout << "\ndone" << endl;
  return 0;
}
