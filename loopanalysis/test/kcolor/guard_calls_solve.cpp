// guard_calls_solve.cpp — the guard function calls back into solve() —
// pass rejects
//
// isSafe() here also invokes solve() itself (a contrived, semantically odd
// but structurally valid shape -- never actually reached at node==N since
// the recursive check is behind an always-false condition, but present in
// the IR either way). checkGuardSideEffects()'s cross-recursion check
// explicitly rejects any call from the guard function back into solve().
//
// Expected: NOT detected.
//
// Extract target: kc_crossrecur_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_crossrecur_solve(int node, vector<int>& color, int m, int N,
                         vector<vector<int>>& graph);

bool kc_crossrecur_isSafe(int node, vector<int>& color,
                          vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  if (node < 0) {
    // Dead code (node is never negative) -- but a real call site to
    // solve() from inside the guard function, statically present.
    return kc_crossrecur_solve(node, color, col, n, graph);
  }
  return true;
}

bool kc_crossrecur_solve(int node, vector<int>& color, int m, int N,
                         vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_crossrecur_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_crossrecur_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_crossrecur_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_crossrecur_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
