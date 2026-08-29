#include <c2cudaq.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

static Graph makeCompleteWeightedGraph(int N, unsigned seed) {
  Graph g;
  g.num_nodes = N;
  mt19937 rng(seed);
  uniform_int_distribution<int> wdist(1, 20);
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
      g.edges.emplace_back(i, j, (double)wdist(rng));
  return g;
}

static int classicalTsp(const Graph& g) {
  int N = g.num_nodes;
  vector<vector<int>> cost(N, vector<int>(N, 0));
  for (auto& [u, v, w] : g.edges) {
    cost[u][v] = (int)w;
    cost[v][u] = (int)w;
  }
  vector<int> nodes;
  for (int i = 1; i < N; i++) nodes.push_back(i);
  int minCost = INT_MAX;
  do {
    int currCost = 0, currNode = 0;
    for (int v : nodes) { currCost += cost[currNode][v]; currNode = v; }
    currCost += cost[currNode][0];
    minCost = min(minCost, currCost);
  } while (next_permutation(nodes.begin(), nodes.end()));
  return minCost;
}

static void section(const char* title) {
  cout << "\n=== " << title << " ===\n" << flush;
}

int main() {
  section("RE-CHECK after qubo_tsp fix: N=3 (9 qubits), layers=2, 10 seeds");
  {
    Graph g = makeCompleteWeightedGraph(3, 1);
    int truth = classicalTsp(g);
    int pass = 0, invalid = 0, total = 10;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_tsp(g, 2, seed);
      if (r.objective == truth) pass++;
      if (r.objective < 0) invalid++;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  truth=" << truth << "  pass=" << pass << "/" << total
         << "  invalid=" << invalid << "/" << total
         << "  (" << secs << "s for " << total << " seeds)\n" << flush;
  }

  section("RE-CHECK after qubo_tsp fix: N=4 (16 qubits), layers=2, 10 seeds");
  {
    Graph g = makeCompleteWeightedGraph(4, 2);
    int truth = classicalTsp(g);
    int pass = 0, invalid = 0, total = 10;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_tsp(g, 2, seed);
      if (r.objective == truth) pass++;
      if (r.objective < 0) invalid++;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  truth=" << truth << "  pass=" << pass << "/" << total
         << "  invalid=" << invalid << "/" << total
         << "  (" << secs << "s for " << total << " seeds)\n" << flush;
  }

  cout << "\ndone\n";
  return 0;
}
