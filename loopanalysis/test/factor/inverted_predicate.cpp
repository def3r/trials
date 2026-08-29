// inverted_predicate.cpp — a*b != n with inverted branch structure —
// still DETECTED
//
// Written as "continue scanning while the product does NOT match, fall
// through to the found-block on equality" -- source-level ICMP_NE with
// swapped branch targets, the opposite of basic.cpp's direct ICMP_EQ.
// Not a REJECT after all: instcombine canonicalizes this back into the
// exact same ICMP_EQ + normal-branch-order shape basic.cpp produces
// (observed directly: `%cmp4.not = icmp eq i32 %mul, %n`, named ".not"
// because instcombine tracked the inversion but still normalized the
// predicate). matchFactor()'s ICMP_EQ-only check (step 6) is therefore
// more robust than it looks in isolation -- canonicalization already
// erases this class of surface-level rewrite before the matcher ever
// runs, so there's no distinct "inverted" IR shape to guard against here.
//
// Expected: DETECTED.

bool factor_inverted(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b != n) {
        continue;
      }
      outA = a;
      outB = b;
      return true;
    }
  }
  outA = 1;
  outB = n;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced factor search loop with call to @factor_impl
// CHECK: call i1 @factor_impl(i32
