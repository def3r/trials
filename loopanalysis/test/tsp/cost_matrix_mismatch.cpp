// cost_matrix_mismatch.cpp — wrap-to-start epilogue reads a DIFFERENT cost
// matrix than the scoring loop — pass rejects
//
// The running accumulation uses `cost[currNode][nodes[i]]`, but the
// wrap-to-start epilogue closes the tour against a second, unrelated matrix
// `cost2[currNode][0]`. matchTsp()'s epilogue matcher (matchDoubleIndexAdd)
// requires the epilogue's first-index container to equal Inner.CostMatrix;
// here Cont1Expected (cost) != cost2's alloca/arg source, so the epilogue is
// never found and the outer match fails.
//
// Expected: NOT detected. Correctly rejected — no CloseAdd found.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_cost_mismatch>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_cost_mismatch(vector<vector<int>>& cost, vector<vector<int>>& cost2) {
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

    // Wrap-to-start close reads a different matrix than the scoring loop.
    currCost += cost2[currNode][0];

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
