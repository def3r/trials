// inner_log.cpp — log_step() called inside maxCliques()'s body — pass
// rejects
//
// Same shape as basic.cpp, but maxCliques() also calls an external logging
// function on every candidate. checkSideEffects() scans maxCliques()'s body
// for calls beyond the recognised structural set (self-call, guard call,
// the clique[size] index call, both std::max calls, std::vector helpers)
// and rejects any other side-effecting call.
//
// Expected: NOT detected.

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

void mc_innerlog_log_step(int v) {
  cout << v << "\n";
}

bool mc_innerlog_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_innerlog_maxCliques(int start, vector<int>& clique, int size, int N,
                           vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    mc_innerlog_log_step(v);  // side effect beyond the recognised shape
    clique[size] = v;
    if (mc_innerlog_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_innerlog_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_innerlog_findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_innerlog_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: clique_impl
