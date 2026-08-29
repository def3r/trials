// both_in.cpp — counts edges where BOTH endpoints are in S, not a cut — pass
// rejects Counts edges where BOTH endpoints are inside S — effectively the
// number of "internal edges" (a measure of subgraph density). This is the
// opposite of cut-edge counting. NOT a MaxCut algorithm.
//
// Expected: should NOT be detected. BUT the current pass HAS THE SAME BUG
// as tn_directed.cpp:
//   Step 1.9 uses `continue` instead of `return std::nullopt` for non-XOR.
//   The AND condition `a_in && b_in` produces `and i1 %ne_a, %ne_b`.
//   This is not XOR, so the loop body does `continue`, then the function
//   falls through and returns a valid ScoringLoopMatch — FALSE POSITIVE.
//
// Concretely, `a_in && b_in` compiles to:
//   %and = and i1 %icmp_ne_a, %icmp_ne_b
// vs MaxCut's XOR:
//   %xor = xor i1 %icmp_ne_a, %icmp_ne_b
//
// Fix needed: same as tn_directed.cpp — require a valid XOR to be found,
// not just fail silently when XOR is absent.
//
// Extract target:
//   llvm-extract
//   -func=_Z22compute_max_int_degreeeSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& vertices);

// Find the subset S maximizing the number of edges with BOTH endpoints in S.
int compute_max_int_degree(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  int best = 0;
  vector<int> best_S;

  for (auto S : partitions) {
    int internal = 0;
    vector<pair<int, int>> int_edges;

    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      // AND condition: both endpoints inside S (NOT XOR)
      if (a_in && b_in) {
        internal++;
        int_edges.push_back({a, b});
      }
    }

    if (internal > best) {
      best = internal;
      best_S = S;
    }
  }

  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
