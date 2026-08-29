// loop_starts_elsewhere.cpp — the candidate loop ignores `start`, always
// begins at 0 — pass rejects
//
// A bug where the loop always scans from vertex 0 regardless of the
// `start` parameter (so it revisits earlier vertices on every recursive
// call -- wrong, but structurally valid IR). matchMaxCliques() step 5
// requires the loop's phi to begin from StartArg specifically
// (LoopPhi->getIncomingValueForBlock(Preheader) == StartArg) -- a check
// with no kcolor equivalent at all, since kcolor's loop bound was a plain,
// search-independent constant. Here the loop's preheader value is the
// literal 0, not the `start` parameter, so this check fails even though
// `start` is still correctly advanced (`v + 1`) in the recursive call
// itself.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_wrongstart_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_wrongstart_maxCliques(int start, vector<int>& clique, int size, int N,
                             vector<vector<int>>& graph) {
  int best = 0;
  for (int v = 0; v < N; v++) {  // ignores `start`, always begins at 0
    clique[size] = v;
    if (mc_wrongstart_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_wrongstart_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_wrongstart_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_wrongstart_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
