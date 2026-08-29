// uint_nodes.cpp — node IDs as unsigned int
// MaxCut with vertex IDs as `unsigned int`. The pass hardcodes the subset
// end() demangled name as "std::vector<int, std::allocator<int>>::end()".
// With vector<unsigned>, the demangled name becomes
// "std::vector<unsigned int, std::allocator<unsigned int>>::end()" which does
// not match → isValidCmp returns false → validXorOp = false → step 1.9 rejects.
//
// Expected: SHOULD detect (semantically identical MaxCut), but MISSES because
// isValidCmp hardcodes the int-specialised vector end() name.
//
// Extract target:
//   llvm-extract
//   -func=_Z20compute_maxcut_uintSt6vectorIjSaIjEES_ISt4pairIjjESaIS3_EE

#include <algorithm>
#include <vector>
using namespace std;

vector<vector<unsigned>> enumerate_subsets_u(vector<unsigned>&);

int compute_maxcut_uint(vector<unsigned> nodes,
                        vector<pair<unsigned, unsigned>> edges) {
  vector<vector<unsigned>> parts = enumerate_subsets_u(nodes);
  int best = 0;
  vector<unsigned> best_S;
  for (auto S : parts) {
    int cut = 0;
    for (auto [u, v] : edges) {
      bool u_in = find(S.begin(), S.end(), u) != S.end();
      bool v_in = find(S.begin(), S.end(), v) != S.end();
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
