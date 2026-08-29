// best_path_output.cpp — also records the winning permutation into an
// output vector — pass rejects
//
// Same TSP algorithm as basic.cpp, but on every new-best tour the current
// permutation is copied into an output parameter `bestPath`, mirroring
// MaxCut's best_S pattern. Unlike MaxCut, tsp_impl has no third argument to
// carry a recovered output back to the caller — checkSideEffects() (gate 2)
// must reject this rather than silently replace the loop and drop the
// `bestPath = nodes;` assignment.
//
// This is a regression test for a real gap that was found while building
// this test suite: gate 2 originally allowed *any* std::vector-named call
// in the outer loop unconditionally, which would have silently dropped this
// assignment. checkSideEffects() now rejects a vector assignment/copy call
// whose `this` pointer doesn't trace back to the permutation container
// itself.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_best_path>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_best_path(vector<vector<int>>& cost, vector<int>& bestPath) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

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
    if (currCost == minCost)
      bestPath = nodes;  // side effect tsp_impl cannot preserve

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
