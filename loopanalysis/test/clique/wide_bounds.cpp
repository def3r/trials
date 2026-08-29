// wide_bounds.cpp — vertex count (N) widened to long long — pass rejects
//
// N is widened; start/clique/size/graph are unchanged. matchMaxCliques()'s
// NArg detection still succeeds structurally (N is found as a distinct
// Argument operand of the loop's bound comparison), but performReplacement()
// emits @clique_impl with a fixed i32 N parameter (matching the bridge's
// actual C++ signature) -- building a CallInst with an i64 argument against
// that i32 parameter would be a type mismatch. matchMaxCliques() explicitly
// checks NArg->getType()->isIntegerTy(32) and rejects here BEFORE
// performReplacement() ever runs, added proactively based on the exact
// crash kcolor's wide_bounds.cpp originally found (see kcolor's
// README.md/clique.md analysis/kcolor/report.html) -- so this test confirms
// the preemptive fix, not a new discovery.
//
// Expected: NOT detected, no crash.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_wide_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_wide_maxCliques(int start, vector<int>& clique, int size, long long N,
                       vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_wide_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_wide_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_wide_findMaxClique(vector<vector<int>>& graph, long long N) {
  vector<int> clique(N, 0);
  return mc_wide_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
