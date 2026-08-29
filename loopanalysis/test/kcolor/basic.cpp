// basic.cpp — canonical m-coloring backtracking search (test/kcolor.cpp's
// shape). Expected: DETECTED by kcolor-pass.
//
// What's the same as the reference:
//   - solve(node, color, m, N, graph): base case node==N, a "try each
//     color 1..m" loop, a guard call (isSafe), assign color[node]=i before
//     the self-recursive call, backtrack color[node]=0 on failure.
//   - graphColoring(graph, m, N): allocates color(N,0), calls
//     solve(0, color, m, N, graph) -- the top-level invocation kcolor-pass
//     replaces.
//
// Extract target: kc_basic_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_basic_isSafe(int node, vector<int>& color, vector<vector<int>>& graph,
                     int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_basic_solve(int node, vector<int>& color, int m, int N,
                    vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_basic_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_basic_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_basic_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_basic_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level solve() call with call to @kcolor_impl
// CHECK: call i1 @kcolor_impl(ptr
// CHECK: declare i1 @kcolor_impl(ptr, i32, i32, ptr)
