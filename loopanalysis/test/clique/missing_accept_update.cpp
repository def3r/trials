// missing_accept_update.cpp — only ONE std::max call, the "accept this
// extension" update is missing — pass rejects
//
// A real, plausible algorithmic bug: forgetting to check whether the
// just-extended clique itself (size + 1) beats the running best, and only
// ever updating from the recursive search's result. matchMaxCliques() step
// 11 requires exactly two std::max<int> call sites; this has one.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_missingaccept_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_missingaccept_maxCliques(int start, vector<int>& clique, int size, int N,
                                vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_missingaccept_isClique(size + 1, clique, graph)) {
      // missing: best = max(best, size + 1);
      best = max(best, mc_missingaccept_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_missingaccept_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_missingaccept_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
