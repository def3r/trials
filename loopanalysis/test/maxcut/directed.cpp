// directed.cpp — directed cut condition (AND not XOR) — pass rejects
// Directed maxcut: counts edges that cross S→V\S in ONE direction only
// (a∈S and b∉S), using an AND condition instead of XOR.
//
// This is NOT MaxCut — it overcounts asymmetric partitions and is a different
// problem (max directed cut / max s-t flow objective).
//
// Expected: should NOT be detected. BUT the current pass HAS A BUG:
//   Step 1.9 checks if the condition is XOR via:
//     if (XorInst->getOpcode() != Instruction::Xor) { continue; }
//   When the condition is an AND (or any non-XOR), the loop just `continue`s
//   without returning nullopt. After the loop exits with no rejection, the
//   function returns a valid ScoringLoopMatch — a FALSE POSITIVE.
//
// The AND condition `a_in && !b_in` compiles to:
//   %not_b = icmp eq ptr %find_b, %end_b     ; !b_in
//   %and   = and i1 %icmp_ne_a, %not_b
// This is an `and i1` instruction, not `xor i1`.
//
// Fix needed: change `continue` to `return std::nullopt` when the condition
// is not a valid XOR of two find-vs-end comparisons.
//
// Extract target:
//   llvm-extract
//   -func=_Z21compute_directed_mcutSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& vertices);

int compute_directed_mcut(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  int best = 0;
  vector<int> best_S;

  for (auto S : partitions) {
    int crossing = 0;
    vector<pair<int, int>> crossing_edges;

    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      // AND condition: only a→b direction (NOT XOR)
      if (a_in && !b_in) {
        crossing++;
        crossing_edges.push_back({a, b});
      }
    }

    if (crossing > best) {
      best = crossing;
      best_S = S;
    }
  }

  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
