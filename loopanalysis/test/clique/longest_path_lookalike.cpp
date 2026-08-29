// longest_path_lookalike.cpp — longest simple path search, NOT max-clique,
// reshaped to fit the same self-recursion pattern matchMaxCliques() looks
// for
//
// A genuinely different problem: extend a path one vertex at a time,
// checking only that the newest vertex is reachable from the previous one
// (a single-edge check), tracking the longest path found. The shape is
// otherwise identical to maxCliques()'s: same start/size roles, the same
// no-backtrack assign, the same two-std::max-call running-best pattern.
// The only semantic difference is the guard function's internal logic
// (single adjacent-pair check instead of all-pairs), and since guard
// functions are matched *opaquely* -- never inspected internally, same as
// isClique/isSafe always have been -- that difference is invisible to the
// matcher.
//
// This DOES currently match and get replaced -- confirmed by running the
// pass, not assumed. matchMaxCliques() has no way to distinguish "extend a
// path checking only the newest edge" from "extend a clique checking all
// pairs": both compile to an identical shape (self-call with the same
// loop-phi/formal-param "+1" split, a no-backtrack assign, a guard call
// taking a derived size+1 argument, two std::max calls forming a
// running-best). If replaced, the pass hands a directed reachability graph
// to c2q_clique as if it were an undirected clique-adjacency matrix and
// answers a completely different question than the one this code asks -- a
// silent wrong-answer bug, not just a missed optimization.
//
// Left as a documented, known limitation (not fixed here), same call as
// kcolor's nqueens_lookalike.cpp: telling "all-pairs constraint check" apart
// from "single-edge constraint check" from IR shape alone, without deeper
// semantic reasoning about what the guard call's second container actually
// represents, is a genuine limit of purely-structural matching.

#include <algorithm>
#include <vector>
using namespace std;

bool mc_lp_isReachable(int size, vector<int>& path, vector<vector<int>>& graph) {
  if (size <= 1) {
    return true;
  }
  return graph[path[size - 2]][path[size - 1]] != 0;
}

int mc_lp_longestPath(int start, vector<int>& path, int size, int N,
                      vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    path[size] = v;
    if (mc_lp_isReachable(size + 1, path, graph)) {
      best = max(best, size + 1);
      best = max(best, mc_lp_longestPath(v + 1, path, size + 1, N, graph));
    }
  }
  return best;
}

int mc_lp_findLongestPath(vector<vector<int>>& graph, int N) {
  vector<int> path(N, 0);
  return mc_lp_longestPath(0, path, 0, N, graph);
}

// --- lit check directives (read by update.py) ---
// Expected: NOT detected -- but XFAIL: currently IS detected (see above).
// XFAIL: *
// CHECK-NOT: clique_impl
