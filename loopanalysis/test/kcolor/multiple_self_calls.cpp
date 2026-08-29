// multiple_self_calls.cpp — solve() has TWO self-recursive call sites —
// pass rejects
//
// A second, decoy self-call after the main loop (never actually reached in
// a real run since it's dead code behind `node == -1`, which never holds --
// irrelevant here since this file is only ever compiled to IR and never
// executed; the point is purely the static call count). matchSolve() step
// 1.1 requires exactly one self-recursive CallBase in the function; finding
// two rejects immediately.
//
// Expected: NOT detected.
//
// Extract target: kc_multiself_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_multiself_isSafe(int node, vector<int>& color,
                         vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_multiself_solve(int node, vector<int>& color, int m, int N,
                        vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_multiself_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_multiself_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  // Decoy second self-call -- dead code, never reached (node is never
  // negative), but statically present.
  if (node == -1) {
    return kc_multiself_solve(node, color, m, N, graph);
  }
  return false;
}

bool kc_multiself_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_multiself_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
