// sum_cut.cpp — sums all cut values across subsets (no max tracking) — pass
// rejects Computes the SUM of cut values across ALL subsets, rather than
// tracking the maximum. The outer accumulator is ADD-based, not max-based.
//
// Expected: NOT detected. Correctly rejected at Phase 2.4:
//   The outer loop has no `icmp sgt(inner_acc, outer_acc)` — the outer
//   accumulator is updated via `add`, so there is no comparison between the
//   inner cut value and the running sum. Phase 2.4 scans all outer-loop
//   blocks for such an icmp and finds nothing.
//
// The outer loop header phis DO pass Phase 2.1 (ptr + i32 init 0),
// and the outer loop condition passes Phase 2.2. Only Phase 2.4 rejects.
//
// Extract target:
//   llvm-extract
//   -func=_Z17compute_total_cutSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& vertices);

int compute_total_cut(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  int total = 0;  // outer accumulator — SUM, not max

  for (auto S : partitions) {
    int crossing = 0;

    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in))
        crossing++;
    }

    // ADD, not max: no comparison, no select/phi merge
    total += crossing;
  }

  return total;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
