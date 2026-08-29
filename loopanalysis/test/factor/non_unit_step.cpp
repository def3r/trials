// non_unit_step.cpp — inner loop increments by 2, not 1 — pass rejects
//
// Skips even candidates for b. matchFactor()'s induction-phi recognition
// (isAddOne(), shared with every other pass in this project) only
// accepts a canonical add-by-1 backedge; a step of 2 fails that check, so
// no inner induction phi is found at all.
//
// Expected: NOT detected.

bool factor_step2(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b += 2) {
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
