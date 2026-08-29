// score_no_max.cpp — inner scoring loop exists but no enclosing max-tracking
// loop — pass rejects The inner scoring loop matches step 1 structurally (ptr
// phi + i32 phi, two finds, XOR gate, add accumulator) but there is NO
// enclosing outer loop. Step 2 calls Inner.L->getParentLoop() which returns
// nullptr → reject.
//
// This tests that a plain cut-scoring helper (no maximization) is not
// misidentified as a full MaxCut implementation.
//
// Expected: NOT detected. Correctly rejected at step 2 (null parent loop).
//
// Extract target:
//   llvm-extract
//   -func=_Z14score_cut_onlySt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

// Compute cut value for one fixed subset S — no outer maximization loop.
int score_cut_only(vector<int> S, vector<pair<int, int>> edges) {
  int cut = 0;
  for (auto [u, v] : edges) {
    bool u_in = find(S.begin(), S.end(), u) != S.end();
    bool v_in = find(S.begin(), S.end(), v) != S.end();
    if (u_in != v_in)
      cut++;
  }
  return cut;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
