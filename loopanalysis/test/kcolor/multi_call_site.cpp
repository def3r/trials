// multi_call_site.cpp — solve(0, ...) invoked from TWO different functions
// — both call sites replaced
//
// Demonstrates that Phase 2 isn't limited to a single "the" entry point:
// findTopLevelCalls() scans ALL of solve's users, so every call site with a
// literal node == 0 argument gets replaced, regardless of which function it
// lives in.
//
// Expected: DETECTED (both call sites).
//
// Extract target: kc_multicall_ (isSafe, solve, graphColoringA, graphColoringB)

#include <vector>
using namespace std;

bool kc_multicall_isSafe(int node, vector<int>& color,
                         vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_multicall_solve(int node, vector<int>& color, int m, int N,
                        vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_multicall_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_multicall_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_multicall_graphColoringA(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  return kc_multicall_solve(0, color, m, N, graph);
}

bool kc_multicall_graphColoringB(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  return kc_multicall_solve(0, color, m, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level solve() call with call to @kcolor_impl
// CHECK: replaced top-level solve() call with call to @kcolor_impl
// CHECK-COUNT-2: call i1 @kcolor_impl(ptr
