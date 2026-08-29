// basic.cpp — canonical max-clique backtracking search (test/clique.cpp's
// shape). Expected: DETECTED by clique-pass.
//
// What's the same as the reference:
//   - isClique(size, clique, graph): all-pairs check over clique[0..size-1].
//   - maxCliques(start, clique, size, N, graph): for v in [start, N), assign
//     clique[size]=v, guard with isClique(size+1, ...), and if safe track
//     best via two running-max updates (accept this extension; keep
//     searching deeper) with no backtrack and no early exit.
//   - findMaxClique(graph, N): allocates clique(N,0), calls
//     maxCliques(0, clique, 0, N, graph) -- the top-level invocation
//     clique-pass replaces.
//
// Extract target: none needed (Module pass, scans the whole file).

#include <algorithm>
#include <vector>
using namespace std;

bool mc_basic_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_basic_maxCliques(int start, vector<int>& clique, int size, int N,
                        vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_basic_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_basic_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_basic_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_basic_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level maxCliques() call with call to @clique_impl
// CHECK: call i32 @clique_impl(ptr
// CHECK: declare i32 @clique_impl(ptr, i32, ptr)
