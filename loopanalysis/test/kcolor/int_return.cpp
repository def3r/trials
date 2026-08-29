// int_return.cpp — solve() returns int (0/1) instead of bool — pass
// rejects
//
// Semantically identical (0/1 used exactly like false/true), but the
// return type is i32 instead of i1. matchSolve() requires
// F.getReturnType()->isIntegerTy(1) -- kcolor_impl's fixed bridge signature
// returns i1, so this is a deliberate scope boundary, not a limitation
// slated to be fixed -- see README.md.
//
// Expected: NOT detected.
//
// Extract target: kc_intreturn_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

int kc_intreturn_isSafe(int node, vector<int>& color,
                        vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return 0;
    }
  }
  return 1;
}

int kc_intreturn_solve(int node, vector<int>& color, int m, int N,
                       vector<vector<int>>& graph) {
  if (node == N) {
    return 1;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_intreturn_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_intreturn_solve(node + 1, color, m, N, graph))
        return 1;
      color[node] = 0;
    }
  }
  return 0;
}

int kc_intreturn_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_intreturn_solve(0, color, m, N, graph))
    return 1;
  return 0;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
