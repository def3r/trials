// assign_backtrack_mismatch.cpp — backtrack writes into a DIFFERENT
// container than the assign store — pass rejects
//
// The assign store writes color[node] = i (identifying ColorArg), but on
// backtrack the code mistakenly clears graph[0][node] instead of
// color[node]. matchSolve() step 1.9 requires the backtrack store's
// container to be the SAME as the assign store's (ColorArg); here it
// traces to GraphArg instead, so BacktrackStore is never found.
//
// Expected: NOT detected.
//
// Extract target: kc_mismatch_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_mismatch_isSafe(int node, vector<int>& color, vector<vector<int>>& graph,
                        int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_mismatch_solve(int node, vector<int>& color, int m, int N,
                       vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_mismatch_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_mismatch_solve(node + 1, color, m, N, graph))
        return true;
      graph[0][node] = 0;  // wrong container -- should be color[node]
    }
  }
  return false;
}

bool kc_mismatch_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_mismatch_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
