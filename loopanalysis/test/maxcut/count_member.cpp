// count_member.cpp — membership test via std::count() instead of std::find()
// MaxCut using `std::count(...) > 0` for membership instead of
// `std::find(...) != end()`. Semantically identical: count > 0 iff element is
// present. Step 1.3 identifies candidate calls by checking the mangled name for
// "find" AND the demangled name for "std::find<". std::count's mangled name is
// _ZSt5count... which does not contain "find" → 0 calls collected → rejected.
//
// Expected: SHOULD detect (semantically identical MaxCut), but MISSES because
// step 1.3 find-detection is tied to the std::find name.
//
// Extract target:
//   llvm-extract
//   -func=_Z20compute_maxcut_cntSt6vectorIiSaIiEES_ISt4pairIiiESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>&);

int compute_maxcut_cnt(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> parts = enumerate_subsets(nodes);
  int best = 0;
  vector<int> best_S;
  for (auto S : parts) {
    int cut = 0;
    for (auto [u, v] : edges) {
      bool u_in = count(S.begin(), S.end(), u) > 0;
      bool v_in = count(S.begin(), S.end(), v) > 0;
      if (u_in != v_in)
        cut++;
    }
    if (cut > best) {
      best = cut;
      best_S = S;
    }
  }
  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced MaxCut loops with call to @maxcut_impl
// CHECK: call i32 @maxcut_impl(ptr
