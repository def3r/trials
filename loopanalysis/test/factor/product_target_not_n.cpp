// product_target_not_n.cpp — a*b compared against n+1, not n — pass
// rejects
//
// Both loop bounds correctly use the raw Argument n, but the product
// check's target is a derived value (n+1), not n itself -- isolates the
// ProdCmp identity check specifically (matchFactor() step 6: the
// non-Mul operand of the eq compare must be the *same* NArg as both loop
// bounds, checked a third time here). A plausible off-by-one bug, not a
// structurally different algorithm.
//
// Expected: NOT detected.

bool factor_offtarget(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n + 1) {
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
