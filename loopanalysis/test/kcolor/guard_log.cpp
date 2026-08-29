// guard_log.cpp — log_check() called inside isSafe()'s body — pass rejects
//
// Same shape as basic.cpp, but the guard function also calls an external
// logging function on every neighbor check. checkGuardSideEffects() scans
// the guard function's body (separately from solve()'s own gate) for calls
// beyond std::vector helpers and rejects any other side-effecting call.
//
// Expected: NOT detected.
//
// Extract target: kc_guardlog_ (isSafe, solve, graphColoring, log_check)

#include <iostream>
#include <vector>
using namespace std;

void kc_guardlog_log_check(int k) {
  cout << k << "\n";
}

bool kc_guardlog_isSafe(int node, vector<int>& color, vector<vector<int>>& graph,
                        int n, int col) {
  for (int k = 0; k < n; k++) {
    kc_guardlog_log_check(k);  // side effect beyond the recognised shape
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_guardlog_solve(int node, vector<int>& color, int m, int N,
                       vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_guardlog_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_guardlog_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_guardlog_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_guardlog_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
