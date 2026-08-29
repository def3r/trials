// no_backtrack.cpp — color[node] is never reset after a failed recursive
// call — pass rejects
//
// A buggy variant that assigns colors on the way down but never undoes the
// assignment on the way back up. This isn't just structurally different --
// it's actually WRONG (later branches at earlier nodes might see a stale
// color and incorrectly reject an otherwise-safe assignment) -- so
// correctly declining to identify it as m-coloring is the right call, not
// just a matcher limitation. matchSolve() step 1.9 requires a store back
// into Container[NodeArg] on the failure edge of the self-call; there
// isn't one here.
//
// Expected: NOT detected.
//
// Extract target: kc_nobacktrack_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_nobacktrack_isSafe(int node, vector<int>& color,
                           vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_nobacktrack_solve(int node, vector<int>& color, int m, int N,
                          vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_nobacktrack_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_nobacktrack_solve(node + 1, color, m, N, graph))
        return true;
      // no backtrack: color[node] stays at the last-tried value
    }
  }
  return false;
}

bool kc_nobacktrack_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_nobacktrack_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
