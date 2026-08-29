// triangular_start.cpp — inner loop starts from b = a, not a constant
// (XFAIL)
//
// A real optimization: no factor pair needs b < a (it would already have
// been found as (b, a) when the outer loop was at that earlier value), so
// skipping symmetric duplicates by starting the inner loop from the
// outer loop's own current value is a legitimate thing to write. But
// matchFactor() requires BPhi's preheader-incoming value to be a
// ConstantInt (step 3); here it's APhi itself, a Value, not a constant.
//
// Same "known limitation, not solved" precedent as clique's
// accumulator_promoted_to_phi.cpp and TSP's min_cmp_form.cpp -- see
// analysis/factor/factor.md §1a, decision 1.
//
// Expected: NOT detected. Known limitation.

bool factor_triangular(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = a; b < n; b++) {
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
// XFAIL: *
// CHECK: factor_impl
