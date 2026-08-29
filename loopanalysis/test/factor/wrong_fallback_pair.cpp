// wrong_fallback_pair.cpp — "not found" edge returns (0, 0), not (1, n) —
// pass rejects
//
// Exercises the "keep it tight" design decision: matchFactor() requires
// the exhausted-search merge phis to carry exactly (ConstantInt 1, NArg),
// not any constant pair. A hand-written (0, 0) sentinel means the same
// thing to a human reader, but isn't the specific pair decode_factors'
// convention expects, so it's rejected by design rather than accepted as
// "any reasonable fallback."
//
// Expected: NOT detected.

bool factor_wrongfallback(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n) {
        outA = a;
        outB = b;
        return true;
      }
    }
  }
  outA = 0;
  outB = 0;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
