// plain_call_site.cpp — top-level call site is a plain `call`, not an
// `invoke` — pass still detects
//
// basic.cpp's findMaxClique() constructs `clique` locally, so its call to
// maxCliques(0, ...) is an `invoke` (clique's destructor needs an unwind
// edge). Here `clique` is passed in by reference from this function's own
// caller instead of being constructed locally, so there's nothing to clean
// up on unwind at this call site -- clang emits a plain `call`. Exercises
// the *other* branch of performReplacement() (the non-invoke path),
// complementing basic.cpp's invoke case.
//
// Expected: DETECTED.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_plaincall_isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int mc_plaincall_maxCliques(int start, vector<int>& clique, int size, int N,
                            vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (mc_plaincall_isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_plaincall_maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

// `clique` is passed by reference, not constructed here -- plain call, no
// unwind edge needed at this call site.
int mc_plaincall_entry(vector<vector<int>>& graph, vector<int>& clique, int N) {
  return mc_plaincall_maxCliques(0, clique, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced top-level maxCliques() call with call to @clique_impl
// CHECK: call i32 @clique_impl(ptr
