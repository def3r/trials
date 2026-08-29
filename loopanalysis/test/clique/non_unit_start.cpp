// non_unit_start.cpp — self-call's `start` slot advances by 2, not 1 —
// pass rejects
//
// The recursive step skips a vertex (`v + 2` instead of `v + 1`).
// matchMaxCliques()'s self-call argument classification only recognises
// `add(LoopPhi, 1)` for the start slot -- the constant operand must be
// exactly one -- so `v + 2` never matches, and no argument gets identified
// as StartArg at all.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_nonunitstart_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_nonunitstart_maxCliques(int start, vector<int>& clique, int size, int N,
                               vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_nonunitstart_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_nonunitstart_maxCliques(v + 2, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_nonunitstart_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_nonunitstart_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
