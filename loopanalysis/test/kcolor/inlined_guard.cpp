// inlined_guard.cpp — safety check inlined directly into solve(), no
// separate helper function — pass rejects
//
// Semantically identical to basic.cpp, but the isSafe() logic is written
// directly inside solve()'s loop body instead of calling a separate
// function. matchSolve() step 1.4 requires the guard condition to be a
// CallBase; here it's the result of an inline loop (a phi/boolean
// computed in-block), so GuardCall is never found.
//
// Expected: NOT detected.
//
// Extract target: kc_inlined_ (solve, graphColoring)

#include <vector>
using namespace std;

bool kc_inlined_solve(int node, vector<int>& color, int m, int N,
                      vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    bool safe = true;
    for (int k = 0; k < N; k++) {
      if (k != node && graph[k][node] == 1 && color[k] == i) {
        safe = false;
        break;
      }
    }
    if (safe) {
      color[node] = i;
      if (kc_inlined_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_inlined_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_inlined_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
