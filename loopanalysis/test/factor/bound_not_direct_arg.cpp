// bound_not_direct_arg.cpp — outer loop bound is a derived local
// (n/2 + 1), not the raw Argument — pass rejects
//
// A real optimization (no factor pair has both members > n/2, so the
// outer loop only needs to search up to there) -- but matchFactor()'s
// outer-bound check requires a direct Argument operand in the icmp (step
// 4), the same NArg-identity requirement every check in this matcher
// relies on. A derived bound, however sensible, isn't a raw Argument.
//
// In the actual compiled IR this trips two things at once, not one:
// instcombine also canonicalizes `a < n/2+1` into `icmp sle %a.0, %div`
// (folding the +1 into the predicate instead of the operand), so the
// ICMP_SLT/ICMP_ULT predicate check (also step 4) rejects it before the
// Argument-identity check even gets evaluated. Either alone would be
// enough; observed together here. Deliberate scope boundary, same
// precedent as clique's wrong_arg_count.cpp (semantically valid,
// unsupported derivation shape) -- not solved here.
//
// Expected: NOT detected.

bool factor_halfbound(int n, int& outA, int& outB) {
  int limit = n / 2 + 1;
  for (int a = 2; a < limit; a++) {
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
