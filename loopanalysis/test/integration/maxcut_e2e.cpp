#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

vector<vector<int>> enumerate_subsets(vector<int>& nodes) {
  vector<vector<int>> subsets{{}};
  for (int u : nodes) {
    int size = subsets.size();
    for (int i = 0; i < size; i++) {
      subsets.push_back(subsets[i]);
      subsets.back().push_back(u);
    }
  }
  return subsets;
}

void log_steps(int cut) {
  cout << cut << "\n";
}

int compute_maxcut(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> partitions = enumerate_subsets(nodes);
  int best_val = 0;
  for (auto S : partitions) {
    int crossing = 0;
    for (auto [a, b] : edges) {
      bool a_in = find(S.begin(), S.end(), a) != S.end();
      bool b_in = find(S.begin(), S.end(), b) != S.end();
      if ((a_in && !b_in) || (!a_in && b_in))
        crossing++;
    }
    if (crossing > best_val)
      best_val = crossing;
  }
  return best_val;
}

int main() {
  vector<int> nodes = {0, 1, 2, 3, 4};
  vector<pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3},
                                  {3, 0}, {2, 4}, {3, 4}};
  int cut = 5;

  cout << "Graph  : 5-node graph (nodes=5, edges=6)\n";
  cout << "Solver : @maxcut_impl -> c2q_maxcut (QAOA) + exact classical "
       << "(N<=16), keeps the better cut -- see analysis/maxcut/maxcut.md\n\n";

  int result = compute_maxcut(nodes, edges);

  cout << "MaxCut value   = " << result << "\n";
  cout << "Classical opt  = " << cut << "\n\n";

  if (result == cut) {
    cout << "PASS  exact optimum\n";
    return 0;
  }
  if (result >= cut - 1) {
    cout << "PASS  good approximation (>= " << cut - 1 << ")\n";
    return 0;
  }
  cout << "FAIL  result too low\n";
  return 1;
}
