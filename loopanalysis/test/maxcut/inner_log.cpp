// inner_log.cpp — log_steps() called inside inner loop — pass rejects
// MaxCut-shaped outer+inner loop structure, but the inner loop also calls
// log_steps() — an external function with side effects (writes to cout).
//
// The pass MUST reject this: replacing the loops with @maxcut_impl would
// silently drop every log_steps() invocation, which is observable behaviour.
//
// Root cause without the fix: checkSideEffects() (gate 2) skips *all* inner
// sub-loop blocks (they are in InnerBlocks), so it never sees log_steps().
// matchScoringLoop() validates only structural properties of the inner loop
// (2 finds, XOR gate, accumulator phi) and did not scan for extra calls.
//
// Fix: matchScoringLoop() now scans inner-loop non-header blocks for calls
// that are not one of the recognised structural calls (FindU, FindV, GetU,
// GetV, or std::vector helpers) and returns nullopt if any such call writes
// to memory.

#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& nodes);

void log_steps(int cut) {
  cout << cut << "\n";
}

int compute_maxcut_log(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);

  int best_val = 0;
  vector<int> best_S;

  for (auto S : partitions) {
    int crossing = 0;

    for (auto [a, b] : edges) {
      // Side effect inside the inner loop — must block transformation.
      log_steps(best_val);
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in))
        crossing++;
    }

    if (crossing > best_val) {
      best_val = crossing;
      best_S = S;
    }
  }

  return best_val;
}

// --- lit check directives (read by update.py) ---
// CHECK-NOT: maxcut_impl
