// assign_wrong_value.cpp — clique[size] stores v + 1, not v — pass rejects
//
// An off-by-one bug: the assign store writes a different vertex than the
// one the loop is actually iterating over. matchMaxCliques() step 9 checks
// the AssignStore's *value* as well as its index -- it must be exactly
// LoopPhi (the loop's own candidate variable), not a derived expression.
// kcolor never checked this (any value stored into color[node] was
// accepted); clique's matcher is stricter here.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_wrongassign_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_wrongassign_maxCliques(int start, vector<int>& clique, int size, int N,
                              vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v + 1;  // should be v
    if (mc_wrongassign_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_wrongassign_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_wrongassign_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_wrongassign_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
