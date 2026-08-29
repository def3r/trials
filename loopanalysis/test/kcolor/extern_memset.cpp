// extern_memset.cpp — memset destination is a global buffer — pass rejects
//
// Looks like m-coloring backtracking but also zeroes an external global
// buffer on every recursive call. solve() is kept at exactly 5 arguments
// (touching the buffer through a global rather than an extra parameter) so
// this test isolates the side-effect gate specifically, rather than
// accidentally tripping the F.arg_size() != 5 check instead -- see
// README.md's note on double-checking which step actually rejects a test.
// The memset destination is a global (not a local alloca), so
// checkSolveSideEffects must reject it -- solve() has an unaccounted side
// effect.
//
// Expected: NOT detected.
//
// Extract target: kc_externmemset_ (isSafe, solve, graphColoring)

#include <cstring>
#include <vector>
using namespace std;

int kc_externmemset_scratch[16];

bool kc_externmemset_isSafe(int node, vector<int>& color,
                            vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_externmemset_solve(int node, vector<int>& color, int m, int N,
                           vector<vector<int>>& graph) {
  // Zero a global buffer -- genuine side effect the pass cannot account
  // for.
  memset(kc_externmemset_scratch, 0, sizeof(kc_externmemset_scratch));

  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_externmemset_isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (kc_externmemset_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_externmemset_graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (kc_externmemset_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: kcolor_impl
