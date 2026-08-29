// swapped_passthrough.cpp — self-call passes N where m should go (and vice
// versa) — pass rejects
//
// The recursive call swaps two of its own arguments relative to solve()'s
// own parameter order. matchSolve() step 1.2 requires every non-recursion
// argument to be threaded through unchanged (SelfCall->getArgOperand(i) ==
// F.getArg(i) for all i other than the node slot); swapping m and N breaks
// that for both positions.
//
// Expected: NOT detected.
//
// Extract target: kc_swapped_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_swapped_isSafe(int node, vector<int>& color,
                       vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_swapped_solve(int node, vector<int>& color, int m, int N,
                      vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_swapped_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      // m and N swapped relative to this function's own parameter order.
      if (kc_swapped_solve(node + 1, color, N, m, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_swapped_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_swapped_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
