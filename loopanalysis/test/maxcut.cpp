#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

// Returns a binary vector (0 or 1) representing the set assignment for each
// node
vector<int> randomized_max_cut(int num_nodes) {
  vector<int> sets(num_nodes);
  srand(time(NULL));

  // The core loop
  for (int i = 0; i < num_nodes; i++) {
    sets[i] = rand() % 2;
  }

  return sets;
}

// graph is represented as an adjacency matrix where graph[i][j] is the edge
// weight
int greedy_max_cut(int num_nodes,
                   const vector<vector<int>>& graph,
                   vector<int>& state) {
  bool improved = true;

  // Outer loop: Keep going until we get stuck in a local minimum
  while (improved) {
    improved = false;

    // Inner loop: Check every node to see if flipping it helps
    for (int i = 0; i < num_nodes; i++) {
      int gain = 0;

      // Calculate the change in cut if we flip node 'i'
      for (int j = 0; j < num_nodes; j++) {
        if (graph[i][j] > 0) {
          // If they are in the SAME set, flipping 'i' ADDS the edge to the cut
          if (state[i] == state[j]) {
            gain += graph[i][j];
          }
          // If they are in DIFFERENT sets, flipping 'i' REMOVES the edge from
          // the cut
          else {
            gain -= graph[i][j];
          }
        }
      }

      // The Conditional Flip
      if (gain > 0) {
        state[i] = 1 - state[i];  // Flip the binary state (0 to 1, or 1 to 0)
        improved = true;
      }
    }
  }

  return 0;  // In a real implementation, you'd calculate and return the final
             // cut value
}

int actual2(vector<int> nodes, vector<pair<int, int>> edges) {
  int cut = 0;

  return cut;
}

int actual(vector<int> nodes, vector<pair<int, int>> edges) {
  int cut = 0;

  vector<vector<int>> subsets{{}};
  for (int u : nodes) {
    int size = subsets.size();
    for (int i = 0; i < size; i++) {
      subsets.push_back(subsets[i]);
      subsets.back().push_back(u);
    }
  }

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

  vector<int> V1{};
  for (int u : nodes) {
    if (find(V0.begin(), V0.end(), u) == V0.end()) {
      V1.push_back(u);
    }
  }

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

  return cut;
}

int main() {
  vector<int> nodes{0, 1, 2, 3};
  vector<pair<int, int>> edges{{0, 1}, {1, 2}, {2, 3}, {3, 0}};
  actual(nodes, edges);
}
