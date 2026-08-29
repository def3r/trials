#include <c2cudaq.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

// Complete graph, symmetric integer weights 1-20, via a fixed seed per N
// so the same instance is reused across seed sweeps and depth sweeps.
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

// Classical brute-force TSP over the same Graph representation, for
// ground truth. Mirrors test/tsp.cpp's tsp()/bridge.cpp's tsp_impl, but
// reads weights from a Graph's edge list instead of a cost matrix.
static int classicalTsp(const Graph& g) {
  int N = g.num_nodes;
  vector<vector<int>> cost(N, vector<int>(N, 0));
  for (auto& [u, v, w] : g.edges) {
    cost[u][v] = (int)w;
    cost[v][u] = (int)w;
  }
  vector<int> nodes;
  for (int i = 1; i < N; i++)
    nodes.push_back(i);
  int minCost = INT_MAX;
  do {
    int currCost = 0, currNode = 0;
    for (int v : nodes) {
      currCost += cost[currNode][v];
      currNode = v;
    }
    currCost += cost[currNode][0];
    minCost = min(minCost, currCost);
  } while (next_permutation(nodes.begin(), nodes.end()));
  return minCost;
}

static void section(const char* title) {
  cout << "\n=== " << title << " ===\n" << flush;
}

int main() {
  // ---- Qubit scaling is the headline finding here: TSP one-hot encodes
  // city x position (N^2 qubits), not city alone (N qubits, like
  // maxcut/kcolor/clique all use). N=6 needs 36 qubits -- already past
  // the 28-qubit simulator ceiling before any completeness/cost question
  // even gets asked. Confirm this directly, not just from the formula. ----
  section("Qubit ceiling: N=6 (36 qubits) should throw immediately");
  try {
    Graph g = makeCompleteWeightedGraph(6, 1);
    auto r = c2q_tsp(g, 2, 0);
    cout << "  UNEXPECTED: did not throw, got objective=" << r.objective << "\n";
  } catch (const exception& e) {
    cout << "  threw as expected: " << e.what() << "\n";
  }

  // ---- Correctness/completeness: the only N values that fit under the
  // 28-qubit ceiling at all are N=3 (9 qubits), N=4 (16 qubits), N=5 (25
  // qubits). Compare against classicalTsp() ground truth, multi-seed
  // where affordable. ----
  section("Completeness: N=3 (9 qubits), layers=2, 10 seeds");
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

  section("Completeness: N=4 (16 qubits), layers=2, 10 seeds");
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

  section("Completeness: N=5 (25 qubits), layers=2, 5 seeds");
  {
    Graph g = makeCompleteWeightedGraph(5, 3);
    int truth = classicalTsp(g);
    int pass = 0, invalid = 0, total = 5;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_tsp(g, 2, seed);
      if (r.objective == truth) pass++;
      if (r.objective < 0) invalid++;
      cout << "    seed=" << seed << "  objective=" << r.objective << "\n" << flush;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  truth=" << truth << "  pass=" << pass << "/" << total
         << "  invalid=" << invalid << "/" << total
         << "  (" << secs << "s for " << total << " seeds, "
         << (secs / total) << "s/seed avg)\n" << flush;
  }

  // ---- Depth: does more QAOA depth rescue completeness at N=5, the
  // ceiling case, the way it helped clique at its own ceiling? Kept to 3
  // seeds given N=5's per-seed cost. ----
  section("Depth sweep: N=5 (25 qubits), 3 seeds per layer count");
  {
    Graph g = makeCompleteWeightedGraph(5, 3);
    int truth = classicalTsp(g);
    for (int layers : {2, 4}) {
      int pass = 0, invalid = 0, total = 3;
      auto t0 = chrono::steady_clock::now();
      for (int seed = 0; seed < total; seed++) {
        auto r = c2q_tsp(g, layers, seed);
        if (r.objective == truth) pass++;
        if (r.objective < 0) invalid++;
      }
      auto t1 = chrono::steady_clock::now();
      double secs = chrono::duration<double>(t1 - t0).count();
      cout << "  layers=" << layers << "  pass=" << pass << "/" << total
           << "  invalid=" << invalid << "/" << total
           << "  (" << secs << "s for " << total << " seeds)\n" << flush;
    }
  }

  cout << "\ndone\n";
  return 0;
}
