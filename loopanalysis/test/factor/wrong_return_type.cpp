// wrong_return_type.cpp — returns int (1/0) instead of bool — pass rejects
//
// Semantically identical search, but the "found" signal is an int return
// instead of a bool. matchFactor() requires F.getReturnType()->isIntegerTy(1)
// -- a scope boundary matching factor_impl's actual i1 return type, not a
// limitation.
//
// Expected: NOT detected.

int factor_retint(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n) {
        outA = a;
        outB = b;
        return 1;
      }
    }
  }
  outA = 1;
  outB = n;
  return 0;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
