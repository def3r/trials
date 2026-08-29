// wrong_arg_count.cpp — an extra, unrelated parameter — pass rejects
//
// Same search, but the function takes 4 parameters instead of 3. This is
// a deliberate scope boundary in matchFactor() (F.arg_size() != 3), not a
// limitation slated to be fixed: the fixed 3-argument signature (n, outA,
// outB) is also what rules out a whole class of lookalike (two different
// bound arguments) by construction -- see product_target_not_n.cpp and
// analysis/factor/factor.md.
//
// Expected: NOT detected.

bool factor_argcount(int n, int unused, int& outA, int& outB) {
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

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
