// basic.cpp — canonical brute-force factor-pair search (test/factor.cpp's
// shape). Expected: DETECTED by factor-pass.
//
// Nested double loop over pairs (a, b) in [2, n) x [2, n), phase-flip
// analog: early-return true the moment a*b == n is found; if the search
// is exhausted, falls back to the (1, n) sentinel pair and returns false.
// This is the classical shape c2q_factor's Grover oracle accelerates --
// see analysis/factor/factor.md §1a for why it's a pair search and not
// trial division.

bool factor_basic(int n, int& outA, int& outB) {
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
// CHECK: replaced factor search loop with call to @factor_impl
// CHECK: call i1 @factor_impl(i32
// CHECK: declare i1 @factor_impl(i32, ptr, ptr)
