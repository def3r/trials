// plain_call_site.cpp — top-level call site is a plain `call`, not an
// `invoke` — pass still detects
//
// basic.cpp's graphColoring() constructs `color` locally, so its call to
// solve(0, ...) is an `invoke` (color's destructor needs an unwind edge).
// Here `color` is passed in by reference from this function's own caller
// instead of being constructed locally, so there's nothing to clean up on
// unwind at this call site -- clang emits a plain `call`. This exercises
// the *other* branch of performReplacement() (the non-invoke path),
// complementing basic.cpp's invoke case.
//
// Expected: DETECTED.
//
// Extract target: kc_plaincall_ (isSafe, solve, entry)

#include <vector>
using namespace std;

bool kc_plaincall_isSafe(int node, vector<int>& color,
                         vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_plaincall_solve(int node, vector<int>& color, int m, int N,
                        vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_plaincall_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_plaincall_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

// `color` is passed by reference, not constructed here -- plain call, no
// unwind edge needed at this call site.
bool kc_plaincall_entry(vector<vector<int>>& graph, vector<int>& color, int m,
                        int N) {
  return kc_plaincall_solve(0, color, m, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level solve() call with call to @kcolor_impl
// CHECK: call i1 @kcolor_impl(ptr
