// score_no_min.cpp — inner scoring loop exists but no enclosing
// permutation/min-tracking loop — pass rejects
//
// The inner loop matches step 1 structurally (idx phi + prev-node phi,
// cost accumulator, cost[u][v] add) but there is no enclosing outer loop —
// this just scores ONE given tour. matchTsp() calls
// Inner.L->getParentLoop(), which returns nullptr, so the match fails.
//
// This tests that a plain tour-scoring helper (no minimisation over
// permutations) is not misidentified as a full TSP implementation.
//
// currCost is `volatile` deliberately: with no std::min call anywhere in
// this function, nothing takes currCost's address, so a plain (non-volatile)
// local gives mem2reg no reason to keep it memory-resident — it gets
// promoted straight to an SSA phi and matchTspScoringLoop() never finds a
// CostAdd at all (see min_cmp_form.cpp for that separate, unrelated failure
// mode). `volatile` keeps the loop in the shape step 1 expects, isolating
// the rejection to "no enclosing loop" specifically.
//
// Expected: NOT detected.
//
// Extract target:
//   llvm-extract -func=<mangled score_tour_only>

#include <vector>
using namespace std;

// Compute the cost of one fixed tour — no outer permutation loop.
int score_tour_only(vector<int>& nodes, vector<vector<int>>& cost) {
  volatile int currCost = 0;
  int currNode = 0;

  for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
    currCost += cost[currNode][nodes[i]];
    currNode = nodes[i];
  }

  currCost += cost[currNode][0];

  return currCost;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: tsp_impl
