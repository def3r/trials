// guard_arg_not_derived.cpp — guard call takes `size` raw instead of
// `size + 1` — pass rejects
//
// isClique() is called with the CURRENT size, not the size after including
// the candidate vertex just written to clique[size] -- so it never actually
// checks the new vertex against the rest of the clique. Structurally valid
// IR, semantically wrong (a real bug someone could write). matchMaxCliques()
// step 8 requires the guard call's relevant argument to be `add(SizeArg,
// 1)`, a derived value; `size` passed raw doesn't match.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_guardraw_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_guardraw_maxCliques(int start, vector<int>& clique, int size, int N,
                           vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_guardraw_isClique(size, clique, graph)) {  // should be size + 1
      best = max(best, size + 1);
      best = max(best, mc_guardraw_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_guardraw_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_guardraw_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
