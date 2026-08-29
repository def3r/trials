// guard_calls_self.cpp — the guard function calls back into maxCliques() —
// pass rejects
//
// isClique() here also invokes maxCliques() itself (contrived, semantically
// odd, but structurally valid -- the call is behind an always-false
// condition so it's never actually reached, but it's a real call site
// statically present in the IR either way). checkGuardSideEffects()'s
// cross-recursion check explicitly rejects any call from the guard
// function back into maxCliques().
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

int mc_crossrecur_maxCliques(int start, vector<int>& clique, int size, int N,
                             vector<vector<int>>& graph);

bool mc_crossrecur_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  if (size < 0) {
    // Dead code (size is never negative) -- but a real call site to
    // maxCliques() from inside the guard function, statically present.
    return mc_crossrecur_maxCliques(0, clique, size, size, graph) != 0;
  }
  return true;
}

int mc_crossrecur_maxCliques(int start, vector<int>& clique, int size, int N,
                             vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_crossrecur_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_crossrecur_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_crossrecur_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_crossrecur_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
