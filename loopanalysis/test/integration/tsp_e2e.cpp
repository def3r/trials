#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>
using namespace std;

// Same shape as test/tsp.cpp / test/tsp/basic.cpp: brute-force TSP via
// std::next_permutation, std::min for the running-best update.

int tsp(vector<vector<int>>& cost) {
  int numNodes = cost.size();
  vector<int> nodes;
  for (int i = 1; i < numNodes; i++)
    nodes.push_back(i);

  int minCost = INT_MAX;

  do {
    int currCost = 0;
    int currNode = 0;

    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
      currCost += cost[currNode][nodes[i]];
      currNode = nodes[i];
    }

    currCost += cost[currNode][0];

    minCost = min(minCost, currCost);

  } while (next_permutation(nodes.begin(), nodes.end()));

  return minCost;
}

int main() {
  vector<vector<int>> cost = {
      {0, 10, 15, 20}, {10, 0, 35, 25}, {15, 35, 0, 30}, {20, 25, 30, 0}};
  int expected = 80;

  cout << "Graph  : 4-city complete graph (test/tsp.cpp's own example)\n";
  cout << "Solver : @tsp_impl -> classical brute force (4 cities is above "
       << "the N<=3 kernel-first cutoff -- see analysis/tsp/tsp.md)\n\n";

  int result = tsp(cost);

  cout << "TSP min cost   = " << result << "\n";
  cout << "Classical opt  = " << expected << "\n\n";

  if (result == expected) {
    cout << "PASS  exact optimum\n";
    return 0;
  }
  cout << "FAIL  result incorrect\n";
  return 1;
}
