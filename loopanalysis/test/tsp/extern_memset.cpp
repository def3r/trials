// extern_memset.cpp — memset destination is a function argument, not a
// local alloca — pass rejects
//
// Looks like TSP but also zeroes an external output buffer on every outer
// iteration. The memset destination is a function argument (not a local
// alloca), so checkSideEffects must reject it — the outer loop has an
// unaccounted side effect.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_extern_memset>

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>
using namespace std;

int tsp_extern_memset(vector<vector<int>>& cost, int* out_scratch, int scratch_len) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  int minCost = INT_MAX;

  do {
    // Zero an external buffer — genuine side effect the pass cannot
    // account for.
    memset(out_scratch, 0, scratch_len * sizeof(int));

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
// CHECK-NOT: tsp_impl
