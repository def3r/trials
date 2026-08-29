// accumulator_promoted_to_phi.cpp — hand-written comparisons instead of
// std::max() (XFAIL)
//
// Same algorithm as basic.cpp, but both running-max updates are written as
// explicit `if (x > best) best = x;` instead of calling std::max(). `best`
// only stays memory-resident because std::max<int>(const int&, const int&)
// takes its arguments by reference; with nothing forcing that escape here,
// sroa/mem2reg promotes `best` straight to an SSA phi. matchMaxCliques()'s
// step 11 (running-max accumulator detection) only recognises the
// std::max<> call form -- with best promoted away, step 1 (self-call
// detection) still succeeds, but there's no memory-resident accumulator
// for the std::max-call scan to find at all.
//
// This is the same phi-promotion gap TSP's min_cmp_form.cpp already
// documents for kcolor-style accumulators, and decision C in
// analysis/clique/clique.md §1d: scoped out deliberately for v1, not an
// oversight.
//
// Expected: NOT detected. Known limitation.

#include <vector>
using namespace std;

bool mc_promoted_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_promoted_maxCliques(int start, vector<int>& clique, int size, int N,
                           vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_promoted_isClique(size + 1, clique, graph)) {
      if (size + 1 > best)
        best = size + 1;
      int r = mc_promoted_maxCliques(v + 1, clique, size + 1, N, graph);
      if (r > best)
        best = r;
    }
  }
  return best;
}

int mc_promoted_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_promoted_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// XFAIL: *
// CHECK: clique_impl
