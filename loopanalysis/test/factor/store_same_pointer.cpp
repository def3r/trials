// store_same_pointer.cpp — both outputs write through the same pointer
// parameter — pass rejects
//
// A buggy/degenerate source (outB is never touched): both the found-a and
// found-b values end up written through outA. Originally written to
// isolate the OutAArg != OutBArg check (step 8) -- but the compiled IR
// shows a different, earlier rejection instead. instcombine only
// partially eliminates the redundant `outA = a; outA = b;` pair: the
// first store survives in its own `if.then` block (never merged into the
// shared exit block the way basic.cpp's single store is), so the found
// edge branches to `if.then`, not directly to the exit block matchFactor
// expects (step 6: BodyBI's true-successor must equal ExitBB). Rejected
// correctly, just one step earlier than planned -- kept as-is since it's
// still a genuine, informative REJECT, not the exact one advertised.
//
// Expected: NOT detected.

bool factor_samestore(int n, int& outA, int& outB) {
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n) {
        outA = a;
        outA = b;
        return true;
      }
    }
  }
  outA = 1;
  outA = n;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
