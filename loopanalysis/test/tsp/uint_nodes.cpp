// uint_nodes.cpp — node ids and costs as unsigned int
// TSP with city ids and matrix costs as `unsigned int` instead of `int`.
// Confidence test: unlike MaxCut, tsp-pass's matchers never hardcode a
// demangled container name tied to a specific element type (no
// `vector<int,...>::end()` style string comparisons), so this is expected
// to detect cleanly rather than expose a limitation.
//
// Expected: DETECTED.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_uint>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_uint(vector<vector<unsigned>>& cost) {
  int numNodes = cost.size();
  vector<unsigned> nodes;
  for (unsigned i = 1; i < static_cast<unsigned>(numNodes); i++)
    nodes.push_back(i);

  unsigned minCost = UINT_MAX;

  do {
    unsigned currCost = 0;
    unsigned currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return static_cast<int>(minCost);
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced TSP loop with call to @tsp_impl
// CHECK: call i32 @tsp_impl(ptr
