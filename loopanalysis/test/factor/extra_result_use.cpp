// extra_result_use.cpp — the found pair is also used outside its store —
// pass rejects
//
// resA/resB are promoted by mem2reg to exactly the same merge-phi shape
// matchFactor() looks for (found-value on one edge, fallback constant on
// the other) -- but here that phi value is used twice: once printed,
// once stored into outA/outB. matchFactor()'s step 9 requires each merge
// phi to have exactly one user (its own store), since performReplacement()
// erases them outright; a second user would be left dangling.
//
// Expected: NOT detected.

#include <iostream>
using namespace std;

bool factor_extrause(int n, int& outA, int& outB) {
  int resA, resB;
  for (int a = 2; a < n; a++) {
    for (int b = 2; b < n; b++) {
      if (a * b == n) {
        resA = a;
        resB = b;
        goto done;
      }
    }
  }
  resA = 1;
  resB = n;
done:
  cout << "a=" << resA << " b=" << resB << endl;
  outA = resA;
  outB = resB;
  return true;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: factor_impl
