// wide_edges.cpp — edge endpoints as pair<long long, long long>
// MaxCut algorithm identical to the reference, but with edge endpoints as
// `long long` instead of `int`. This is still a valid MaxCut — the graph
// structure is the same, just using 64-bit node IDs.
//
// Expected: SHOULD be detected as MaxCut but IS NOT.
// Failure point: Step 1.7 checks the inner latch GEP stride:
//   if (!Idx || !Idx->equalsInt(8)) continue;
// pair<int,int> has sizeof = 8, but pair<long long, long long> has sizeof = 16.
// The latch GEP advances by 16 bytes, not 8, so Step 1.7 rejects it.
//
// Similarly, if the node IDs were long long (changing the subset vector from
// vector<int> to vector<long long>), the outer latch stride would also be wrong
// (std::find would search vector<long long> not vector<int>, and the vector
// element type would differ). But in this version nodes stay int, so only the
// inner stride is affected.
//
// Fix needed: derive stride from the GEP's source element type instead of
// hardcoding the value 8. Or verify the type is pair<int,int> and accept the
// appropriate stride for whatever pair type is found.
//
// Extract target (note: ll suffix in mangled name due to long long params):
//   llvm-extract
//   -func=_Z20compute_maxcut_llSt6vectorIiSaIiEES_ISt4pairIxiESaIS3_EE (actual
//   mangled name — verify with: grep 'define.*compute_maxcut_ll'
//   fn_long_edges.ll)

#include <algorithm>
#include <vector>
using namespace std;

// Subset enumeration still over int nodes
vector<vector<int>> enumerate_subsets(vector<int>& vertices);

// Edge endpoints are long long — same MaxCut algorithm, wider types
int compute_maxcut_ll(vector<int> nodes,
                      vector<pair<long long, long long>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  int best = 0;
  vector<pair<long long, long long>> best_edges;
  vector<int> best_S;

  for (auto S : partitions) {
    int crossing = 0;
    vector<pair<long long, long long>> crossing_edges;

    for (auto [a, b] : edges) {
      // nodes are int but edges are long long — find in int subset using cast
      bool a_in = find(S.begin(), S.end(), (int)a) != S.end();
      bool b_in = find(S.begin(), S.end(), (int)b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in)) {
        crossing++;
        crossing_edges.push_back({a, b});
      }
    }

    if (crossing > best) {
      best = crossing;
      best_edges = crossing_edges;
      best_S = S;
    }
  }

  return best;
}

// --- lit check directives (read by update.py) ---
// CHECK: replaced MaxCut loops with call to @maxcut_impl
// CHECK: call i32 @maxcut_impl(ptr
