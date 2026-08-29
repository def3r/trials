// sibling_function.cpp — a matching search alongside an unrelated
// function in the same file — expected: DETECTED (the matching one only)
//
// factor-pass is a plain FunctionPass, so per-function independence is
// mostly guaranteed by LLVM's own pass-manager architecture rather than
// anything specific to this matcher -- included anyway as a build/lit
// sanity check that an unrelated sibling function in the same translation
// unit doesn't confuse compilation or matching.
//
// Expected: DETECTED (factor_sibling_search only).

int factor_sibling_unrelated(int x, int y) {
  return x + y;
}

bool factor_sibling_search(int n, int& outA, int& outB) {
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
