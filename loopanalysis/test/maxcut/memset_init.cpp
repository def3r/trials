// memset_init.cpp — local vector re-init folds to llvm.memset after instcombine
// MaxCut with a local vector<pair<int,int>> accumulating crossing edges inside
// the outer loop body.  After sroa + instcombine the three null-pointer stores
// that initialise that vector collapse into a single llvm.memset intrinsic.
// The pass must allow memset-to-local-alloca (a store equivalent) and still
// fire.  Regression test for the checkSideEffects memset fix.
//
// Extract target function for IR analysis:
//   llvm-extract
//   -func=_Z14compute_maxcutSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& nodes);

int compute_maxcut(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  int best_val = 0;
  vector<int> best_S;

  for (auto S : partitions) {
    int crossing = 0;
    // This local vector gets zero-initialised each iteration.
    // After instcombine the three null stores collapse to llvm.memset.
    vector<pair<int, int>> crossing_edges;

    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in)) {
        crossing++;
        crossing_edges.push_back({a, b});
      }
    }

    if (crossing > best_val) {
      best_val = crossing;
      best_S = S;
    }
  }

  return best_val;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced MaxCut loops with call to @maxcut_impl
// CHECK: call i32 @maxcut_impl(ptr
// CHECK: declare i32 @maxcut_impl(ptr, ptr, ptr)
