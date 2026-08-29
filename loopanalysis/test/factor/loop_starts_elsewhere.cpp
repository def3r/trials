// loop_starts_elsewhere.cpp — extra code between the outer header and the
// inner loop's entry — pass rejects
//
// A logging call sits between "a is in range" succeeding and entering the
// inner loop, so the inner loop's unique external predecessor is that
// logging block, not the outer header itself. matchFactor() requires
// InnerPreheader == OuterHeader (step 1) -- mirrors TSP's Phase 2 "scoring
// loop's preheader is the permutation loop's header" coupling check, and
// clique's loop_starts_elsewhere.cpp.
//
// Expected: NOT detected.

#include <iostream>
using namespace std;

bool factor_elsewhere(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    cout << "trying a=" << a << endl;
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

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
