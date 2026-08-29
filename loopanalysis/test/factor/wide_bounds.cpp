// wide_bounds.cpp — n is long long (i64), not int (i32) — pass rejects
//
// performReplacement() emits @factor_impl with a fixed i32 N parameter,
// matching the bridge's actual C++ signature -- same reasoning as
// kcolor_pass.cpp's wide_bounds fix and clique_pass.cpp's NArg type
// check. Building a CallInst against a mismatched i64 argument would be
// wrong, so this is rejected rather than silently truncated.
//
// Expected: NOT detected.

bool factor_wide(long long n, long long& outA, long long& outB) {
  for (long long a = 2; a < n; a++) {
    for (long long b = 2; b < n; b++) {
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
