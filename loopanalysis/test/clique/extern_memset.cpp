// extern_memset.cpp — memset destination is a global buffer — pass rejects
//
// Looks like max-clique backtracking but also zeroes a global scratch
// buffer on every recursive call. maxCliques() is kept at exactly 5
// arguments (touching the buffer through a global rather than an extra
// parameter) so this test isolates the side-effect gate specifically,
// rather than accidentally tripping the F.arg_size() != 5 check instead —
// see README.md's note on double-checking which step actually rejects a
// test (the same pitfall that hit kcolor's extern_memset during
// development). The memset destination is a global (not a local alloca),
// so checkSideEffects must reject it.
//
// Expected: NOT detected.

#include <algorithm>
#include <cstring>
#include <vector>
using namespace std;

int mc_externmemset_scratch[16];

bool mc_externmemset_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_externmemset_maxCliques(int start, vector<int>& clique, int size, int N,
                               vector<vector<int>>& graph) {
  // Zero a global buffer -- genuine side effect the pass cannot account
  // for.
  memset(mc_externmemset_scratch, 0, sizeof(mc_externmemset_scratch));

  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_externmemset_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_externmemset_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_externmemset_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_externmemset_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
