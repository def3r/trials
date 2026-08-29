// multi_call_site.cpp — maxCliques(0, ..., 0, ...) invoked from TWO
// different functions — both call sites replaced
//
// Demonstrates that Phase 2 isn't limited to a single "the" entry point:
// findTopLevelCalls() scans ALL of maxCliques's users, so every call site
// with literal start==0 AND size==0 arguments gets replaced, regardless of
// which function it lives in.
//
// Expected: DETECTED (both call sites).

#include <algorithm>
#include <vector>
using namespace std;

bool mc_multicall_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_multicall_maxCliques(int start, vector<int>& clique, int size, int N,
                            vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_multicall_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_multicall_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int mc_multicall_findMaxCliqueA(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_multicall_maxCliques(0, clique, 0, N, graph);
}

int mc_multicall_findMaxCliqueB(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return mc_multicall_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level maxCliques() call with call to @clique_impl
// CHECK: replaced top-level maxCliques() call with call to @clique_impl
// CHECK-COUNT-2: call i32 @clique_impl(ptr
