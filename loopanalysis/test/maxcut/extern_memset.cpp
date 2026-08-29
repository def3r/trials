// extern_memset.cpp — memset destination is a function argument, not a local
// alloca — pass rejects
#include <algorithm>
#include <cstring>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& nodes);

// Looks like MaxCut but also zeroes an external output array each iteration.
// The memset destination is a function argument (not a local alloca) so the
// pass must reject it — the loop has an unaccounted side effect.
int compute_maxcut_ext(vector<int> nodes,
                       vector<pair<int, int>> edges,
                       int* out_scratch,
                       int scratch_len) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);
  int best_val = 0;
  for (auto S : partitions) {
    // Zero an external buffer — genuine side effect the pass cannot account
    // for.
    memset(out_scratch, 0, scratch_len * sizeof(int));
    int crossing = 0;
    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in))
        crossing++;
    }
    if (crossing > best_val)
      best_val = crossing;
  }
  return best_val;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
