// wrong_return_type.cpp — maxCliques() returns bool instead of int — pass
// rejects
//
// A variant that only reports whether any clique of a target size exists,
// not the maximum size found. matchMaxCliques() requires
// F.getReturnType()->isIntegerTy(32) -- clique_impl's fixed bridge
// signature returns i32, so this is a deliberate scope boundary, not a
// limitation slated to be fixed -- see README.md (mirrors kcolor's
// int_return.cpp, but inverted: kcolor requires i1 and rejects i32; this
// pass requires i32 and rejects i1).
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_boolreturn_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

bool mc_boolreturn_maxCliques(int start, vector<int>& clique, int size, int N,
                              vector<vector<int>>& graph) {
  bool best = false;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_boolreturn_isClique(size + 1, clique, graph)) {
      best = best || true;
      best = best || mc_boolreturn_maxCliques(v + 1, clique, size + 1, N, graph);
    }
  }
  return best;
}

bool mc_boolreturn_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_boolreturn_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
