// uint_graph.cpp — vertex ids and matrix entries as unsigned int
// Confidence test: matchMaxCliques() never hardcodes a demangled container
// name tied to a specific element type, so this is expected to detect
// cleanly rather than expose a limitation -- same conclusion as kcolor's
// uint_graph.
//
// Expected: DETECTED.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_uint_isClique(int size, vector<unsigned>& clique, vector<vector<unsigned>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_uint_maxCliques(int start, vector<unsigned>& clique, int size, int N,
                       vector<vector<unsigned>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = (unsigned)v;
    if (mc_uint_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_uint_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_uint_findMaxClique(vector<vector<unsigned>>& graph, int N) {
  vector<unsigned> clique(N, 0);
  return mc_uint_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level maxCliques() call with call to @clique_impl
// CHECK: call i32 @clique_impl(ptr
