// inner_log.cpp — a debug print inside the loop body — pass rejects
//
// Same mechanism as clique/kcolor's inner_log.cpp: the search itself
// matches matchFactor() cleanly (the print doesn't disturb the phi/icmp/
// mul shape), but checkSideEffects() scans every block in the matched
// loop nest for calls beyond the recognised structural instructions, and
// rejects this one. Distinct failure point from most of this suite --
// this is a case that DOES match structurally and is rejected at the
// side-effect gate afterward, not during matchFactor() itself.
//
// Expected: NOT detected.

#include <iostream>
using namespace std;

bool factor_innerlog(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      cout << "trying " << a << "," << b << endl;
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

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
