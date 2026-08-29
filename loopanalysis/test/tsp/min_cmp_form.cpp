// min_cmp_form.cpp — hand-written comparison instead of std::min() (XFAIL)
//
// Same TSP algorithm as basic.cpp, but the min update is written as an
// explicit `if (currCost < minCost) minCost = currCost;` instead of calling
// std::min(). The intent was to exercise matchTsp()'s MinCmpForm path
// (icmp slt/sgt + select/branch store); instead it reveals a deeper miss.
//
// tsp_pass.cpp's own header comment documents the assumption: "TSP's
// cost/min accumulators have their address taken (passed by reference into
// std::min), so they remain memory-resident (alloca + load/store) ... The
// matchers below work on that load-add-store shape instead of phi
// backedges." That assumption only holds because std::min<int> takes its
// arguments by const&, forcing currCost/minCost's addresses to escape and
// blocking mem2reg promotion. Without that call, NOTHING takes their
// address, so sroa+mem2reg promotes both accumulators straight to SSA phis
// — the very shape the header comment says TSP does *not* use. matchTsp()
// has no phi-based fallback (unlike MaxCut's matchMaxCut, which explicitly
// handles phi-merge/select/smax forms), so matchTspScoringLoop() never even
// finds a CostAdd (step 1.3 requires `dyn_cast<AllocaInst>` on the store's
// pointer operand) and the whole match fails before MinCmpForm is ever
// reached.
//
// Expected: NOT detected. Known limitation — matchTsp()'s MinCmpForm path
// is effectively unreachable via natural compilation; TSP written without
// std::min entirely bypasses the memory-resident shape the matcher expects.
//
// Extract target:
//   llvm-extract -func=<mangled tsp_cmp_form>

#include <algorithm>
#include <climits>
#include <vector>
using namespace std;

int tsp_cmp_form(vector<vector<int>>& cost) {
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

    if (currCost < minCost)  // explicit compare, not std::min()
      minCost = currCost;

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

// --- lit check directives (read by update.py) ---
// XFAIL: *
// CHECK: tsp_impl
