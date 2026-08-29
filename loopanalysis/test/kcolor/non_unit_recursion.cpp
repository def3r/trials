// non_unit_recursion.cpp — self-call advances node by 2, not 1 — pass
// rejects
//
// The recursive step skips a node (`node + 2` instead of `node + 1`).
// matchSolve() step 1.2 only recognises `add(FormalArg, 1)` -- the constant
// operand must be exactly one -- so `node + 2` never matches, and no
// argument gets identified as NodeArg at all.
//
// Expected: NOT detected.
//
// Extract target: kc_nonunit_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_nonunit_isSafe(int node, vector<int>& color,
                       vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_nonunit_solve(int node, vector<int>& color, int m, int N,
                      vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_nonunit_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_nonunit_solve(node + 2, color, m, N, graph))  // skips a node
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_nonunit_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_nonunit_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
