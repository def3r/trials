#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

vector<vector<int>> setsubs(vector<int>& nodes) {
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

vector<int> notin(vector<int>& V0, vector<int>& nodes) {
  vector<int> V1{};
  for (int u : nodes) {
    if (find(V0.begin(), V0.end(), u) == V0.end()) {
      V1.push_back(u);
    }
  }
  return V1;
}

void printres(int maxcut_val,
              vector<pair<int, int>>& maxcut_edges,
              vector<int>& V0,
              const vector<int>& V1) {
  cout << "Maxcut val: " << maxcut_val << endl;
  cout << "Edges: " << endl;
  for (auto [u, v] : maxcut_edges) {
    cout << "    " << u << " - " << v << endl;
  }
  cout << "Sets: " << endl;
  for (int u : V0) {
    cout << u << " ";
  }
  cout << endl;
  for (int u : V1) {
    cout << u << " ";
  }
}

int actual(vector<int> nodes, vector<pair<int, int>> edges) {
  vector<vector<int>> subsets = setsubs(nodes);

  int maxcut_val = 0;
  vector<pair<int, int>> maxcut_edges{};
  vector<int> V0{};
  for (auto subset : subsets) {
    int subset_cut_val = 0;
    vector<pair<int, int>> subset_cut_edges{};
    for (auto [u, v] : edges) {
      bool u_in_subset = find(subset.begin(), subset.end(), u) != subset.end();
      bool v_in_subset = find(subset.begin(), subset.end(), v) != subset.end();
      if ((u_in_subset && !v_in_subset) || (!u_in_subset && v_in_subset)) {
        subset_cut_val += 1;
        subset_cut_edges.push_back({u, v});
      }
    }
    if (subset_cut_val > maxcut_val) {
      maxcut_val = subset_cut_val;
      maxcut_edges = subset_cut_edges;
      V0 = subset;
    }
  }

  return maxcut_val;
}

int main() {
  vector<int> nodes{0, 1, 2, 3};
  vector<pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}, {3, 0}};
  actual(nodes, edges);
}
