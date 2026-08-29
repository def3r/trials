// inner_log.cpp — log_step() called inside the inner scoring loop — pass
// rejects
//
// TSP-shaped outer+inner loop structure, but the inner loop also calls
// log_step() — an external function with side effects (writes to cout).
//
// The pass MUST reject this: replacing the loops with @tsp_impl would
// silently drop every log_step() invocation, which is observable behaviour.
// matchTspScoringLoop() step 1.7 scans inner-loop non-header blocks for
// calls that are not one of the recognised structural calls (Index1,
// Index2, PermIndexA, PermIndexB) and rejects if any such call writes to
// memory.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_inner_log>

#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>
using namespace std;

void log_step(int cost) {
  cout << cost << "\n";
}

int tsp_inner_log(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  int minCost = INT_MAX;

  do {
    int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      // Side effect inside the inner loop — must block transformation.
      log_step(currCost);
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
