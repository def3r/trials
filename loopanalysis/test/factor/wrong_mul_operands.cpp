// wrong_mul_operands.cpp — a*a == n (perfect-square search) instead of
// a*b == n — pass rejects
//
// A very plausible near-miss: someone searching for a perfect square root
// of n reuses the same nested-loop skeleton, but the multiply's operands
// are both the outer phi. matchFactor() requires Mul's operands to be
// exactly {APhi, BPhi} in either order (step 6) -- {APhi, APhi} doesn't
// qualify, so this correctly does not match a genuine pair search.
//
// Expected: NOT detected.

bool factor_square(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * a == n) {
        outA = a;
        outB = a;
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
