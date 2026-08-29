// wrong_base_case.cpp — base case uses >= instead of == — pass rejects
//
// `node >= N` is semantically equivalent to `node == N` here (node only
// ever increases by exactly 1, so it can never overshoot N), but
// matchSolve() step 1.6 only recognises ICMP_EQ specifically. `>=`
// compiles to `icmp sge`, which isn't, so BaseCaseCmp/NArg are never
// identified.
//
// Expected: NOT detected.
//
// Extract target: kc_wrongbase_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_wrongbase_isSafe(int node, vector<int>& color,
                         vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_wrongbase_solve(int node, vector<int>& color, int m, int N,
                        vector<vector<int>>& graph) {
  if (node >= N) {  // >= instead of ==
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_wrongbase_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_wrongbase_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_wrongbase_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_wrongbase_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
