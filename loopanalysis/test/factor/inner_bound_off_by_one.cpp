// inner_bound_off_by_one.cpp — inner loop uses <= instead of < — pass
// rejects
//
// An off-by-one typo: the outer loop correctly bounds with `a < n`, but
// the inner loop uses `b <= n`. matchFactor() only accepts ICMP_SLT/
// ICMP_ULT for both loop bounds (step 5). In the compiled IR instcombine
// canonicalizes `<=` into an inverted `icmp sgt` with swapped branch
// targets (`%cmp2.not = icmp sgt i32 %b.0, %n`, same ".not" convention
// seen in inverted_predicate.cpp) rather than leaving a literal SLE --
// ICMP_SGT isn't SLT/ULT either, so it's still rejected, just not via
// the exact predicate value this comment originally implied.
//
// Expected: NOT detected.

bool factor_offbyone(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b <= n; b++) {
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
