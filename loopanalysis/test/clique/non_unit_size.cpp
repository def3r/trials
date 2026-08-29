// non_unit_size.cpp — self-call's `size` slot advances by 2, not 1 — pass
// rejects
//
// Unlike non_unit_start.cpp (which breaks the loop-phi-based "+1" check),
// this breaks the *other* independent "+1" check: the `size` slot must be
// `add(FormalArg, 1)`. `size + 2` never matches, so SizeArg is never
// identified. Splitting this from non_unit_start.cpp matters specifically
// for this pass, since clique has two independent "+1" sources to get
// wrong where kcolor only had one.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_nonunitsize_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_nonunitsize_maxCliques(int start, vector<int>& clique, int size, int N,
                              vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_nonunitsize_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_nonunitsize_maxCliques(v + 1, clique, size + 2, N, graph));
    }
  }
  return best;
}

int mc_nonunitsize_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_nonunitsize_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
