// wide_bounds.cpp — max-colors bound (m) widened to long long — pass
// rejects
//
// Only `m` is widened; node/N/color/graph are unchanged. matchSolve()'s
// structural match for MArg still succeeds cleanly (m itself appears as a
// direct Argument operand of the loop bound comparison, regardless of the
// sext inserted on the loop counter's side) -- but this was originally
// found via a crash, not a clean rejection: performReplacement() builds
// @kcolor_impl with a fixed i32/i32 signature for m/N (matching the
// bridge's actual C++ signature), so an i64 MArg produced a CallInst whose
// argument type didn't match its declared parameter type, tripping an
// LLVM assertion. matchSolve() now explicitly requires MArg/NArg to be
// i32, turning that crash into a clean rejection.
//
// Expected: NOT detected.
//
// Extract target: kc_wide_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_wide_isSafe(int node, vector<int>& color, vector<vector<int>>& graph,
                    int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_wide_solve(int node, vector<int>& color, long long m, int N,
                   vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_wide_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_wide_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_wide_graphColoring(vector<vector<int>>& graph, long long m, int N) {
  vector<int> color(N, 0);
  if (kc_wide_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
