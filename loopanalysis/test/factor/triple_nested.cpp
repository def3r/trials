// triple_nested.cpp — three levels of loop nesting — pass rejects
//
// A spurious innermost loop (bounded by a runtime value, so it can't be
// folded away by simplifycfg/instcombine) sits inside the pair search.
// matchFactor() requires the candidate inner loop to itself have no
// subloops (!InnerL->getSubLoops().empty()); with a third level present,
// this fails cleanly.
//
// Expected: NOT detected.

bool factor_triple(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      for (int k = 0; k < b; k++) {
        if (a * b == n && k == 0) {
          outA = a;
          outB = b;
          return true;
        }
      }
    }
  }
  outA = 1;
  outB = n;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
