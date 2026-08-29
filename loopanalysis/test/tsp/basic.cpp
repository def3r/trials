// basic.cpp — canonical TSP algorithm (brute-force permutation search)
// Same algorithm as the reference (test/tsp.cpp), renamed. Expected:
// DETECTED by tsp-pass.
//
// What's the same as reference:
//   - Outer do-while loop: iterate all permutations via std::next_permutation,
//     track minimum tour cost
//   - Inner loop: walk the permutation, accumulate currCost via
//     cost[currNode][nodes[i]], track currNode
//   - Wrap-to-start epilogue: currCost += cost[currNode][0]
//   - std::min(minCost, currCost) call form for the min update
//
// Extract target function for IR analysis:
//   llvm-extract -func=<mangled tsp_basic>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_basic(vector<vector<int>>& cost) {
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

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced TSP loop with call to @tsp_impl
// CHECK: call i32 @tsp_impl(ptr
// CHECK: declare i32 @tsp_impl(ptr, ptr)
