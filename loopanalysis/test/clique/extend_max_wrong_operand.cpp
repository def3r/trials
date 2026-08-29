// extend_max_wrong_operand.cpp — the "extend" max update maxes against `v`,
// not the self-call's result — pass rejects
//
// The recursive call is still made (so the self-call itself is found
// structurally), but its result is discarded, and best is instead extended
// with the loop's own candidate vertex `v` -- nonsensical, but structurally
// valid IR. matchMaxCliques() step 11 classifies each std::max update by
// its "other" operand: `size + 1` for the accept update, or a value
// identical to the self-call itself for the extend update. `v` matches
// neither, so ExtendMaxCall is never set and the match fails.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_extendwrong_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_extendwrong_maxCliques(int start, vector<int>& clique, int size, int N,
                              vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_extendwrong_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      mc_extendwrong_maxCliques(v + 1, clique, size + 1, N, graph);  // result discarded
      best = max(best, v);  // extends with v, not the self-call's result
    }
  }
  return best;
}

int mc_extendwrong_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_extendwrong_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
