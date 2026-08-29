// no_top_level_call.cpp — solve() matches structurally, but is never
// invoked with node == 0 anywhere — no replacement (score_no_min /
// score_no_max analog)
//
// solve()/isSafe() have the full recognised shape, but nothing in this
// translation unit ever calls solve(0, ...) -- solve's only caller is
// itself (the recursive step). matchSolve() succeeds (Phase 1), but
// findTopLevelCalls() (Phase 2) finds nothing to replace: this is not a
// rejection, it's "matched, nothing to do here."
//
// Expected: NOT detected (no call site to replace).
//
// Extract target: kc_notoplevel_ (isSafe, solve)

#include <vector>
using namespace std;

bool kc_notoplevel_isSafe(int node, vector<int>& color,
                          vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_notoplevel_solve(int node, vector<int>& color, int m, int N,
                         vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_notoplevel_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_notoplevel_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

// No caller anywhere -- solve's only user is its own recursive step.

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
