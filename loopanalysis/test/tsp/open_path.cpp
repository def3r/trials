// open_path.cpp — Hamiltonian PATH (no wrap-to-start close) — pass rejects
//
// Same brute-force permutation search, but this solves the open-path
// variant: minimise the cost of visiting every city once, WITHOUT
// returning to the start. The mandatory wrap-to-start epilogue
// (`currCost += cost[currNode][0]`) is absent — matchTsp() step 2.4
// requires that epilogue (matchDoubleIndexAdd against Inner.CostAcc /
// Inner.PrevNodePhi / Inner.CostMatrix) and returns nullopt when it can't
// find it.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_open_path>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_open_path(vector<vector<int>>& cost) {
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

    // No wrap-to-start close — this is an open path, not a cycle.

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
