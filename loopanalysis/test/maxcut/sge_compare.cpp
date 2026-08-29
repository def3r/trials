// sge_compare.cpp — max comparison uses >= (sge) instead of > (sgt) — pass
// rejects MaxCut where ties go to the LAST visited subset (>= instead of >).
// SGE semantics commit to a specific output (the last tied subset wins) that is
// outside the pass's target: the pass identifies algorithms where any best
// subset is acceptable, not ones with a defined tie-breaking rule.
//
// Expected: SHOULD NOT be detected. Correctly rejected by step 2.4 because
// `cut >= best` compiles to `icmp sge` / `icmp slt` — neither is ICMP_SGT
// and no smax intrinsic is emitted → MaxCompare = nullptr → match fails.
//
// Extract target:
//   llvm-extract
//   -func=_Z21compute_maxcut_sgeSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>&);

int compute_maxcut_sge(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> parts = enumerate_subsets(nodes);
  int best = 0;
  vector<int> best_S;
  for (auto S : parts) {
    int cut = 0;
    for (auto [u, v] : edges) {
      bool u_in = find(S.begin(), S.end(), u) != S.end();
      bool v_in = find(S.begin(), S.end(), v) != S.end();
      if (u_in != v_in)
        cut++;
    }
    if (cut >= best) {  // >= keeps last tied subset; > would keep first
      best = cut;
      best_S = S;
    }
  }
  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
