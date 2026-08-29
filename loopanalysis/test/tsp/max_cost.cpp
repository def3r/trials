// max_cost.cpp — tracks the MAXIMUM tour cost via std::max(), not the
// minimum (longest-path variant) — pass rejects
//
// Structurally identical to the reference TSP except the outer update calls
// std::max() instead of std::min() to track the maximum tour cost. This
// (like basic.cpp's std::min call) keeps maxCost/currCost address-taken and
// memory-resident, so the inner scoring loop still matches step 1 cleanly —
// this isolates the rejection to the outer min-update check specifically,
// rather than being a byproduct of mem2reg promoting everything to phis
// (see min_cmp_form.cpp for that separate, unrelated failure mode).
//
// matchTsp() step 2.5 only recognises a call whose demangled name contains
// "std::min<" (MinCallForm) for the call-based path, and falls back to
// scanning for `icmp slt`/`icmp sgt` (MinCmpForm) otherwise. std::max<int>
// is a real, non-inlined library call (built with -fno-inline): its
// demangled name contains "std::max<", not "std::min<", so MinCallForm
// never matches; and because the comparison happens inside std::max's own
// (uninlined) body, no icmp appears in the caller at all, so MinCmpForm
// finds nothing either. Both forms fail, and the match is rejected.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_max_cost>

#include <algorithm>
#include <vector>
using namespace std;

int tsp_max_cost(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  int maxCost = 0;

  do {
    int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    // std::max, not std::min — tracks the longest tour, not the shortest.
    maxCost = max(maxCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return maxCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
