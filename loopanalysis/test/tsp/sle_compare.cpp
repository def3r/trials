// sle_compare.cpp — min update uses <= (sle) instead of < (slt) — pass
// rejects
//
// TSP where ties go to the LAST visited permutation (<= instead of <). SLE
// semantics commit to a specific tie-breaking rule that is outside the
// pass's target: it identifies algorithms where any minimal tour is
// acceptable, not ones with a defined tie-breaking rule.
//
// currCost/minCost are declared `volatile` here deliberately: a plain
// (non-volatile) manual comparison gives mem2reg no reason to keep either
// accumulator memory-resident (nothing takes their address, unlike
// std::min's by-reference parameters), so it promotes them straight to SSA
// phis and matchTspScoringLoop() never finds a CostAdd at all — see
// min_cmp_form.cpp, which documents that separate, unrelated failure mode.
// `volatile` forces every access through a real load/store (mem2reg never
// promotes volatile allocas), keeping the inner scoring loop in the shape
// step 1 expects, so this isolates the rejection to the outer min-update
// predicate check specifically.
//
// Expected: NOT detected. matchTsp()'s min-update matcher only recognises
// ICMP_SLT (cost < min) or ICMP_SGT (min > cost); `currCost <= minCost`
// compiles to `icmp sle`, which is neither, so MinCmpForm stays null.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_sle>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_sle(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  volatile int minCost = INT_MAX;

  do {
    volatile int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    if (currCost <= minCost)  // <= keeps the last tied tour; < would keep the first
      minCost = currCost;

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
