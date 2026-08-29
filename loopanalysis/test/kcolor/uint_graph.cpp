// uint_graph.cpp — node ids and matrix entries as unsigned int
// Confidence test: matchSolve() never hardcodes a demangled container name
// tied to a specific element type (no vector<int,...> string comparisons
// anywhere), so this is expected to detect cleanly rather than expose a
// limitation -- unlike MaxCut's uint_nodes, which was a genuine miss.
//
// Expected: DETECTED.
//
// Extract target: kc_uint_ (isSafe, solve, graphColoring)

#include <vector>
using namespace std;

bool kc_uint_isSafe(int node, vector<unsigned>& color,
                    vector<vector<unsigned>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && (int)color[k] == col) {
      return false;
    }
  }
  return true;
}

bool kc_uint_solve(int node, vector<unsigned>& color, int m, int N,
                   vector<vector<unsigned>>& graph) {
  if (node == N) {
    return true;
  }
  for (int i = 1; i <= m; i++) {
    if (kc_uint_isSafe(node, color, graph, N, i)) {
      color[node] = (unsigned)i;
      if (kc_uint_solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool kc_uint_graphColoring(vector<vector<unsigned>>& graph, int m, int N) {
  vector<unsigned> color(N, 0);
  if (kc_uint_solve(0, color, m, N, graph))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level solve() call with call to @kcolor_impl
// CHECK: call i1 @kcolor_impl(ptr
