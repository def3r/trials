// exit_log.cpp — a debug print in the shared merge block — pass rejects
//
// checkSideEffects() only scans OuterL->blocks() -- the merge/exit block
// sits outside the loop and isn't covered by it at all. matchFactor()'s
// own step 10 fills that gap: the merge block must contain nothing beyond
// the two recognised phis, the two stores, and the terminating ret.
// Constructed with `goto` so the print genuinely lands in the same shared
// block both control-flow edges reach (a plain if/return doesn't produce
// that -- the print would land in the loop body instead, exercising
// inner_log.cpp's check, not this one).
//
// Expected: NOT detected.

#include <iostream>
using namespace std;

bool factor_exitlog(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n) {
        outA = a;
        outB = b;
        goto done;
      }
    }
  }
  outA = 1;
  outB = n;
done:
  cout << "done" << endl;
  return true;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
