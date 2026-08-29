// guard_log.cpp — log_check() called inside isClique()'s body — pass
// rejects
//
// Same shape as basic.cpp, but the guard function also calls an external
// logging function on every pair check. checkGuardSideEffects() scans the
// guard function's body (separately from maxCliques()'s own gate) for calls
// beyond std::vector helpers and rejects any other side-effecting call.
//
// Expected: NOT detected.

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

void mc_guardlog_log_check(int i, int j) {
  cout << i << " " << j << "\n";
}

bool mc_guardlog_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      mc_guardlog_log_check(i, j);  // side effect beyond the recognised shape
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_guardlog_maxCliques(int start, vector<int>& clique, int size, int N,
                           vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_guardlog_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_guardlog_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_guardlog_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_guardlog_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
