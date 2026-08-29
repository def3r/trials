// weighted.cpp — crossing edges contribute weight 2 instead of 1
// MaxCut where each crossing edge contributes weight 2 instead of 1.
// `cut += 2` compiles to add(AccPhi, 2). Step 1.6's isIncrement() accepts only
// ConstantInt(1) or ZExtInst-from-i1; ConstantInt(2) fails both branches.
// In the branchless path LLVM may emit `shl(zext(xor), 1)` or `mul(zext, 2)`,
// neither of which is a bare ZExtInst → same rejection.
//
// Expected: SHOULD detect (semantically MaxCut with uniform weight 2), but
// MISSES because step 1.6 isIncrement only recognises unit increments.
//
// Extract target:
//   llvm-extract
//   -func=_Z20compute_maxcut_w2St6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>&);

int compute_maxcut_w2(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> parts = enumerate_subsets(nodes);
  int best = 0;
  vector<int> best_S;
  for (auto S : parts) {
    int cut = 0;
    for (auto [u, v] : edges) {
      bool u_in = find(S.begin(), S.end(), u) != S.end();
      bool v_in = find(S.begin(), S.end(), v) != S.end();
      if (u_in != v_in)
        cut += 2;  // uniform weight 2 per crossing edge
    }
    if (cut > best) {
      best = cut;
      best_S = S;
    }
  }
  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced MaxCut loops with call to @maxcut_impl
// CHECK: call i32 @maxcut_impl(ptr
