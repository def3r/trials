// val_only.cpp — MaxCut with value-only tracking (no best_S output)
// Same MaxCut algorithm but WITHOUT the vector<int> best_S assignment in the
// outer update block. This causes instcombine/simplifycfg to fold the empty
// if-then block into a `select` instruction rather than leaving a phi merge.
//
// Expected: SHOULD be detected as MaxCut but IS NOT.
// Failure point: Phase 2.4 BFS looks for a PHINode with incoming values
//   (CutLCSSA, MaxPhi). When the update folds to:
//     %best.new = select i1 %cmp, i32 %cut, i32 %best
//   the outer header phi gets %best.new as its latch value — not CutLCSSA
//   or MaxPhi directly — so HasCut and HasMax are both false and the BFS
//   finds no MaxUpdatePhi.
//
// Fix needed: handle `select i1 icmp_sgt, CutVal, MaxVal` as MaxUpdatePhi.
//
// Extract target:
//   llvm-extract
//   -func=_Z16compute_maxcut_vSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& vertices);

// MaxCut: only track the best value, not which subset achieved it.
// No vector copy-assignment in the update path → branch gets folded to select.
int compute_maxcut_v(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  int best = 0;

  for (auto S : partitions) {
    int cut = 0;

    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in))
        cut++;
    }

    // No vector assignment here — the empty if-then collapses to a select.
    if (cut > best)
      best = cut;
  }

  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced MaxCut loops with call to @maxcut_impl
// CHECK: call i32 @maxcut_impl(ptr
// CHECK: declare i32 @maxcut_impl(ptr, ptr, ptr)
