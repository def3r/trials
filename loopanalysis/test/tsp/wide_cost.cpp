// wide_cost.cpp — cost matrix entries are `long long`, accumulated into an
// `int` running cost — pass rejects
//
// TSP algorithm identical to the reference, but the cost matrix stores
// 64-bit values while currCost/minCost stay `int` (matching the function's
// i32 return type). The implicit narrowing conversion on
// `currCost += cost[currNode][nodes[i]]` inserts a `trunc i64 ... to i32`
// between the cost-matrix load and the add. matchTspScoringLoop() step 1.3
// requires the add's non-accumulator operand to be a bare LoadInst
// (`dyn_cast<LoadInst>(Other)`); here it's a TruncInst wrapping the load, so
// CostAdd is never found.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_wide_cost>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_wide_cost(vector<vector<long long>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  int minCost = INT_MAX;

  do {
    int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];  // i64 -> i32 narrowing
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
