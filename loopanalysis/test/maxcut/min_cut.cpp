// min_cut.cpp — tracks minimum crossing count, not maximum — pass rejects
// Min-cut by brute-force subset enumeration. Structurally identical to the
// reference MaxCut except the outer comparison tracks the MINIMUM crossing
// count, not the maximum.
//
// Expected: NOT detected. Correctly rejected at:
//   Phase 2.1 — MaxPhi init check: initialized to edges.size() (not 0)
//   Phase 2.4 — icmp sgt check: the update comparison is icmp slt
//
// NOTE: both rejections fire independently. Even if init were 0 (which would
// be semantically wrong for min-cut since no subset can have a negative cut),
// Phase 2.4 would still reject because the comparison uses slt not sgt.
//
// Extract target:
//   llvm-extract -func=_Z14compute_mincutSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& vertices);

int compute_mincut(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  // Initialize to upper bound (number of edges), not 0.
  // This also triggers Phase 2.1 rejection (init ≠ ConstantInt(0)).
  int min_val = (int)edges.size();
  vector<int> min_S;

  for (auto S : partitions) {
    int crossing = 0;
    vector<pair<int, int>> crossing_edges;

    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in))
        crossing++;
    }

    // icmp slt — Phase 2.4 specifically requires icmp sgt.
    if (crossing < min_val) {
      min_val = crossing;
      min_S = S;
    }
  }

  return min_val;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
