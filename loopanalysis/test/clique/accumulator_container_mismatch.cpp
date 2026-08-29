// accumulator_container_mismatch.cpp — the two std::max updates target
// DIFFERENT accumulators — pass rejects
//
// A bug where the "accept this extension" and "keep searching deeper"
// updates each maintain their own separate running best instead of a
// shared one. matchMaxCliques() step 11 requires both std::max call sites
// to update the SAME memory-resident accumulator; here they update best1
// and best2 independently. (The final combine outside the loop uses a
// manual ternary, not std::max, specifically so this test isolates the
// "same accumulator" check rather than also tripping the "exactly two
// std::max calls" check via a third call -- see extra_max_call.cpp for
// that one.)
//
// Expected: NOT detected.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_accmismatch_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_accmismatch_maxCliques(int start, vector<int>& clique, int size, int N,
                              vector<vector<int>>& graph) {
  int best1 = 0, best2 = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_accmismatch_isClique(size + 1, clique, graph)) {
      best1 = max(best1, size + 1);
      best2 = max(best2, mc_accmismatch_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best1 > best2 ? best1 : best2;
}

int mc_accmismatch_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_accmismatch_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
