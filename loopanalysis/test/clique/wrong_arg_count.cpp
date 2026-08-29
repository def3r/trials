// wrong_arg_count.cpp — N derived from graph.size() instead of passed as a
// parameter — pass rejects
//
// Semantically identical algorithm, but maxCliques()/isClique() take one
// fewer argument (N is derived internally via graph.size() instead of
// being threaded through as its own parameter). This is a deliberate scope
// boundary in matchMaxCliques() (F.arg_size() != 5), not a limitation
// slated to be fixed -- see README.md.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_argcount_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_argcount_maxCliques(int start, vector<int>& clique, int size,
                           vector<vector<int>>& graph) {
  int N = graph.size();
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_argcount_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_argcount_maxCliques(v + 1, clique, size + 1, graph));
    }
  }
  return best;
}

int mc_argcount_findMaxClique(vector<vector<int>>& graph) {
  vector<int> clique(graph.size(), 0);
  return mc_argcount_maxCliques(0, clique, 0, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
