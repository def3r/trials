// perm_container_mismatch.cpp — std::next_permutation walks a DIFFERENT
// container than the one the scoring loop indexes — pass rejects
//
// The scoring loop reads city order from `nodes`, but the outer loop's
// termination condition permutes a separate (unrelated) vector `shadow`
// instead. matchTsp() step 2.3 requires next_permutation's begin/end to
// trace back to Inner.PermContainer; here they trace to `shadow`, so the
// match fails.
//
// Expected: NOT detected. Correctly rejected — NextPermCall's container
// doesn't match the scoring loop's permutation container.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_perm_mismatch>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_perm_mismatch(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);
  vector<int> shadow = nodes;

  int minCost = INT_MAX;

  do {
    int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    minCost = min(minCost, currCost);

    // Permutes `shadow`, not `nodes` — container mismatch.
  } while (next_permutation(shadow.begin(), shadow.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
