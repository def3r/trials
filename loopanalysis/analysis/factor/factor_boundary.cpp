#include <c2cudaq.h>
#include <chrono>
#include <iostream>
using namespace std;
int main() {
  try {
    auto t0 = chrono::steady_clock::now();
    auto [a, b] = c2q_factor(128);
    auto t1 = chrono::steady_clock::now();
    cout << "n=128: did NOT throw -- a=" << a << " b=" << b
         << " time=" << chrono::duration<double>(t1-t0).count() << "s" << endl;
  } catch (const exception& e) {
    cout << "n=128: threw as expected: " << e.what() << endl;
  }
  return 0;
}
