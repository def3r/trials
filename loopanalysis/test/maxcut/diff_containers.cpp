// diff_containers.cpp — two finds search different containers — pass rejects
// Two find calls search DIFFERENT containers: u is looked up in S (the current
// subset alloca) while v is looked up in `right` (a separate by-value argument
// alloca). Step 1.4 requires stripToContainerSource(find_u_arg0) ==
// stripToContainerSource(find_v_arg0); here S alloca != right alloca → reject.
//
// Expected: NOT detected. Correctly rejected at step 1.4 (ContU != ContV).
//
// Extract target:
//   llvm-extract
//   -func=_Z15compute_bip_cutSt6vectorIiSaIiEES1_S_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>&);

// Bipartite cut: u searched in the chosen left subset S, v searched in the
// fixed right side. Two different source containers → fails step 1.4.
int compute_bip_cut(vector<int> left,
                    vector<int> right,
                    vector<pair<int, int>> edges) {
  vector<vector<int>> left_subs = enumerate_subsets(left);
  int best = 0;
  for (auto S : left_subs) {
    int cut = 0;
    for (auto [u, v] : edges) {
      bool u_in = find(S.begin(), S.end(), u) != S.end();
      bool v_in = find(right.begin(), right.end(), v) != right.end();
      if (u_in != v_in)
        cut++;
    }
    if (cut > best)
      best = cut;
  }
  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
