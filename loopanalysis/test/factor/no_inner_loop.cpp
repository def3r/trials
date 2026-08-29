// no_inner_loop.cpp — one flat loop (div/mod decomposed index) instead of
// a nested pair of loops — pass rejects
//
// Same search space, but walked with a single loop over a flattened index
// instead of two nested loops. matchFactor() requires OuterL to have
// exactly one subloop (OuterL->getSubLoops().size() != 1); a flat loop
// has zero, so this fails at the very first structural check. Nobody
// writes factor search this way from scratch -- see analysis/factor/
// factor.md §1a, decision 4.
//
// Expected: NOT detected.

bool factor_flat(int n, int& outA, int& outB) {
  int span = n > 2 ? n - 2 : 1;
  for (int idx = 0; idx < span * span; idx++) {
    int a = 2 + idx / span;
    int b = 2 + idx % span;
    if (a * b == n) {
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
// CHECK-NOT: factor_impl
