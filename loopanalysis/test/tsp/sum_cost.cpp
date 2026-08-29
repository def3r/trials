// sum_cost.cpp — sums tour costs across ALL permutations, no min tracking —
// pass rejects
//
// Computes the SUM of tour costs across every permutation, rather than the
// minimum. The outer accumulator is ADD-based: no comparison exists between
// the per-tour cost and the running total, so matchTsp()'s min-update
// matcher (step 2.5) finds neither a std::min call nor an icmp slt/sgt and
// the match fails.
//
// currCost/totalCost are `volatile` deliberately: a plain manual add gives
// mem2reg no reason to keep either accumulator memory-resident (nothing
// takes their address, unlike std::min's by-reference parameters), so it
// promotes them straight to SSA phis and matchTspScoringLoop() never finds
// a CostAdd at all — see min_cmp_form.cpp, which documents that separate,
// unrelated failure mode. `volatile` keeps the inner scoring loop in the
// shape step 1 expects, isolating the rejection to the outer min-update
// check specifically.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_sum_cost>

#include <algorithm>
#include <vector>
using namespace std;

int tsp_sum_cost(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  volatile int totalCost = 0;  // outer accumulator — SUM, not min

  do {
    volatile int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    // ADD, not min: no comparison, no select/phi merge.
    totalCost += currCost;

  } while (next_permutation(nodes.begin(), nodes.end()));

  return totalCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
