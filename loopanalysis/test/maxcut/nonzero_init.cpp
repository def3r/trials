// nonzero_init.cpp — outer max accumulator initialised to -1 sentinel (XFAIL)
// MaxCut where the outer max accumulator `best` is initialised to -1 as a
// pessimistic sentinel (meaning "no valid cut found yet"). Step 2.1 checks
// that the i32 phi coming from the preheader is zero (Init->isZero()); -1
// fails that check → outer loop rejected → whole match fails.
//
// Expected: SHOULD detect (semantically valid MaxCut), but MISSES because
// step 2.1 only accepts zero initialisation.
//
// Extract target:
//   llvm-extract
//   -func=_Z24compute_maxcut_sentinelSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>&);

int compute_maxcut_sentinel(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> parts = enumerate_subsets(nodes);
  int best = -1;  // pessimistic: no subset evaluated yet
  vector<int> best_S;
  for (auto S : parts) {
    int cut = 0;
    for (auto [u, v] : edges) {
      bool u_in = find(S.begin(), S.end(), u) != S.end();
      bool v_in = find(S.begin(), S.end(), v) != S.end();
      if (u_in != v_in)
        cut++;
    }
    if (cut > best) {
      best = cut;
      best_S = S;
    }
  }
  return best;
}

// --- lit check directives (read by update.py) ---
// XFAIL: *
// CHECK: maxcut_impl
