// no_top_level_call.cpp — maxCliques() matches structurally, but is never
// invoked with start==0, size==0 anywhere — no replacement
//
// maxCliques()/isClique() have the full recognised shape, but nothing in
// this translation unit ever calls maxCliques(0, ..., 0, ...) --
// maxCliques's only caller is itself (the recursive step). matchMaxCliques()
// succeeds (Phase 1), but findTopLevelCalls() (Phase 2) finds nothing to
// replace: this is not a rejection, it's "matched, nothing to do here."
//
// Expected: NOT detected (no call site to replace).

#include <algorithm>
#include <vector>
using namespace std;

bool mc_notoplevel_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_notoplevel_maxCliques(int start, vector<int>& clique, int size, int N,
                             vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_notoplevel_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_notoplevel_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

// No caller anywhere -- maxCliques's only user is its own recursive step.

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
