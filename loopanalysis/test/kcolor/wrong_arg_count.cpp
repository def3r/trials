// wrong_arg_count.cpp — N derived from graph.size() instead of passed as a
// parameter — pass rejects
//
// Semantically identical algorithm, but solve()/isSafe() take one fewer
// argument (N is derived internally via graph.size() instead of being
// threaded through as its own parameter). This is a deliberate scope
// boundary in matchSolve() (F.arg_size() != 5), not a limitation slated to
// be fixed -- see README.md.
//
// Expected: NOT detected.
//
// Extract target: kc_argcount_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_argcount_isSafe(int node, vector<int>& color,
                        vector<vector<int>>& graph, int col) {
  int n = graph.size();
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_argcount_solve(int node, vector<int>& color, int m,
                       vector<vector<int>>& graph) {
  int N = graph.size();
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_argcount_isSafe(node, color, graph, i)) {
      color[node] = i;
      if (kc_argcount_solve(node + 1, color, m, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_argcount_graphColoring(vector<vector<int>>& graph, int m) {
  vector<int> color(graph.size(), 0);
  if (kc_argcount_solve(0, color, m, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
