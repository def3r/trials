// inner_log.cpp — log_step() called inside solve()'s body — pass rejects
//
// Same shape as basic.cpp, but solve() also calls an external logging
// function on every candidate-color attempt. checkSolveSideEffects() scans
// solve()'s body for calls beyond the recognised structural set (self-call,
// guard call, the two color[node] index calls, std::vector helpers) and
// rejects any other side-effecting call.
//
// Expected: NOT detected.
//
// Extract target: kc_innerlog_ (isSafe, solve, graphColoring, log_step)

#include <iostream>
#include <vector>
using namespace std;

void kc_innerlog_log_step(int node) {
  cout << node << "\n";
}

bool kc_innerlog_isSafe(int node, vector<int>& color, vector<vector<int>>& graph,
                        int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_innerlog_solve(int node, vector<int>& color, int m, int N,
                       vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    kc_innerlog_log_step(node);  // side effect beyond the recognised shape
    if (kc_innerlog_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_innerlog_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_innerlog_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_innerlog_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
