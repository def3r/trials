// scaled_cost.cpp — running cost adds 2x the matrix entry — pass rejects
//
// Each step accumulates `2 * cost[currNode][nodes[i]]` instead of the raw
// matrix entry. The argmin over permutations is unaffected by a uniform
// scale factor, but the returned VALUE would be wrong if replaced with
// @tsp_impl (which computes the unscaled cost) — this must be rejected, not
// just "structurally different". matchTspScoringLoop() step 1.3 requires
// the add's non-accumulator operand to be a bare load of cost[u][v]; here
// it's `mul(load, 2)`, so CostAdd is never found.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_scaled>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_scaled(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  int minCost = INT_MAX;

  do {
    int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += 2 * cost[currNode][nodes[i]];  // scaled, not raw, cost
      currNode = nodes[i];
    }

    currCost += 2 * cost[currNode][0];

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
