// manual_permutation.cpp — hand-rolled permutation stepping instead of
// std::next_permutation — pass rejects
//
// Same scoring/min-tracking shape as the reference TSP, but the outer
// loop's backedge condition is a plain integer counter (`iter < totalIters`)
// with a hand-written swap standing in for permutation advancement, rather
// than a call to std::next_permutation. matchTsp() step 2.3 requires the
// outer latch's branch condition to be a CallBase whose demangled name
// contains "std::next_permutation<"; here it's an icmp on a loop counter, so
// dyn_cast<CallBase> fails and NextPermCall stays null.
//
// (The swap below doesn't enumerate permutations correctly — that's
// irrelevant here; only the IR shape of the outer loop's exit condition
// matters for this test.)
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_manual_perm>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_manual_perm(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  int minCost = INT_MAX;
  int iter = 0;
  int totalIters = numNodes;

  do {
    int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    minCost = min(minCost, currCost);

    // Hand-rolled step — no std::next_permutation call at all.
    if (nodes.size() >= 2) {
      int tmp = nodes[0];
      nodes[0] = nodes[1];
      nodes[1] = tmp;
    }
    iter++;

  } while (iter < totalIters);

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
