// swapped_passthrough.cpp — self-call passes graph.size() instead of N —
// pass rejects
//
// The recursive call recomputes the bound from `graph` instead of passing
// `N` straight through. matchMaxCliques()'s self-call argument
// classification requires every non-start/size argument to equal
// F.getArg(i) exactly (SelfCall->getArgOperand(i) == F.getArg(i)); a call
// to size() is a different SSA value even though it may hold the same
// runtime value, so this fails. (A literal swap with `graph` isn't
// possible here -- different types, wouldn't compile -- and something
// like `N + 0` would just get instcombined back to `N` directly, erasing
// the distinction before the pass ever sees it; graph.size() survives
// canonicalization as a genuinely separate expression.)
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_swapped_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_swapped_maxCliques(int start, vector<int>& clique, int size, int N,
                          vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_swapped_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_swapped_maxCliques(v + 1, clique, size + 1,
                                             (int)graph.size(), graph));
    }
  }
  return best;
}

int mc_swapped_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_swapped_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
