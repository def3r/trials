// no_recursion.cpp — iterative (stack-based) backtracking, no self-call at
// all — pass rejects
//
// Same algorithm, but implemented as an explicit loop with a manual stack
// instead of recursion. There is no self-recursive CallBase anywhere in
// this function, so matchSolve() step 1.1 (SelfCalls.size() != 1) rejects
// before looking at anything else. (This shape also naturally has only 4
// arguments -- no `node` parameter, since recursion depth is a local
// variable instead -- so the arg_size() != 5 guard would reject it too;
// either is a legitimate reason this doesn't match.)
//
// Expected: NOT detected.
//
// Extract target: kc_norecur_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_norecur_isSafe(int node, vector<int>& color,
                       vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_norecur_solve(vector<int>& color, int m, int N,
                      vector<vector<int>>& graph) {
  int node = 0;
  while (node >= 0 && node < N) {
    bool advanced = false;
    for (int i = color[node] + 1; i <= m; i++) {
      if (kc_norecur_isSafe(node, color, graph, N, i)) {
        color[node] = i;
        node++;
        advanced = true;
        break;
      }
    }
    if (!advanced) {
      color[node] = 0;
      node--;
    }
  }
  return node == N;
}

bool kc_norecur_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_norecur_solve(color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
