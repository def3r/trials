// wrong_close_index.cpp — wrap-to-start closes to a non-constant node — pass
// rejects
//
// A wrap-around epilogue IS present (unlike open_path.cpp, which has none
// at all), but it closes back to `nodes[0]` — the first city in the current
// permutation — instead of the literal fixed start (constant index 0). This
// is a distinct bug: it produces a cycle, but not one that returns to the
// same fixed starting city on every permutation.
//
// matchDoubleIndexAdd()'s second-index matcher (used for the epilogue)
// requires a ConstantInt second index (Cont1Expected[FirstIdx][0]); here the
// second index is a runtime load of nodes[0], not a ConstantInt, so the
// epilogue is never recognised and matchTsp() step 2.4 rejects.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_wrong_close>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_wrong_close(vector<vector<int>>& cost) {
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

    // Closes to nodes[0] (runtime value), not the fixed literal start.
    currCost += cost[currNode][nodes[0]];

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
