// extra_max_call.cpp — a third, unrelated std::max call inside the
// candidate loop — pass rejects
//
// Some unrelated bookkeeping (tracking the largest vertex id seen) also
// uses std::max, inside the same loop. matchMaxCliques() step 11 requires
// exactly two std::max<int> call sites within the candidate loop; this has
// three, so MaxCandidates.size() != 2 and the match fails -- the mirror
// image of missing_accept_update.cpp.
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_extramax_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_extramax_maxCliques(int start, vector<int>& clique, int size, int N,
                           vector<vector<int>>& graph) {
  int best = 0;
  int largestSeen = 0;
  for (int v = start; v < N; v++) {
    largestSeen = max(largestSeen, v);  // unrelated third std::max call
    clique[size] = v;
    if (mc_extramax_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_extramax_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best > largestSeen ? best : largestSeen;
}

int mc_extramax_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_extramax_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
