// three_finds.cpp — inner loop has 3 membership checks (hyperedge) — pass
// rejects Inner loop has THREE std::find calls (one per endpoint of a 3-uniform
// hyperedge). Step 1.3 collects all std::find calls and requires exactly 2;
// finding a third causes Finds.size() > 2 → reject (or the final != 2 check).
//
// Expected: NOT detected. Correctly rejected at step 1.3 (3 finds != 2).
//
// Extract target:
//   llvm-extract -func=_Z19compute_hyper_cutSt6vectorIiSaIiEES_I5Edge3SaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>&);

struct Edge3 {
  int u, v, w;
};

// Maximize 3-hyperedges with exactly 1 or 2 endpoints in the chosen subset.
// Three find calls per inner iteration → step 1.3 rejects.
int compute_hyper_cut(vector<int> nodes, vector<Edge3> hedges) {
  vector<vector<int>> parts = enumerate_subsets(nodes);
  int best = 0;
  for (auto S : parts) {
    int cut = 0;
    for (auto& e : hedges) {
      bool u_in = find(S.begin(), S.end(), e.u) != S.end();
      bool v_in = find(S.begin(), S.end(), e.v) != S.end();
      bool w_in = find(S.begin(), S.end(), e.w) != S.end();
      int cnt = (int)u_in + (int)v_in + (int)w_in;
      if (cnt == 1 || cnt == 2)
        cut++;
    }
    if (cut > best)
      best = cut;
  }
  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
