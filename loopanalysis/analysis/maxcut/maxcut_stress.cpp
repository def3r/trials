#include <c2cudaq.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

static Graph makeCompleteGraph(int N) {
  Graph g;
  g.num_nodes = N;
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
      g.edges.emplace_back(i, j, 1.0);
  return g;
}

static Graph makeCycleGraph(int N) {
  Graph g;
  g.num_nodes = N;
  for (int i = 0; i < N; i++)
    g.edges.emplace_back(i, (i + 1) % N, 1.0);
  return g;
}

// Sparse Erdos-Renyi random graph, density ~0.4, integer weights 1-5.
static Graph makeRandomGraph(int N, unsigned seed) {
  Graph g;
  g.num_nodes = N;
  mt19937 rng(seed);
  uniform_real_distribution<double> unif(0.0, 1.0);
  uniform_int_distribution<int> wdist(1, 5);
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++)
      if (unif(rng) < 0.4)
        g.edges.emplace_back(i, j, (double)wdist(rng));
  return g;
}

// Exhaustive classical max cut: try every 2-partition (2^(N-1) distinct,
// since a partition and its complement give the same cut), return best.
static int classicalMaxCut(const Graph& g) {
  int N = g.num_nodes;
  int best = 0;
  for (unsigned mask = 0; mask < (1u << N); mask++) {
    int cut = 0;
    for (auto& [u, v, w] : g.edges)
      if (((mask >> u) & 1) != ((mask >> v) & 1))
        cut += (int)w;
    best = max(best, cut);
  }
  return best;
}

static int analyticalCompleteGraphCut(int N) {
  // Balanced bipartition maximizes cuts in a complete graph.
  int a = N / 2, b = N - a;
  return a * b;
}

static int analyticalCycleCut(int N) {
  // Even cycle: perfectly 2-colorable, cut = N. Odd cycle: one edge must
  // stay uncut, cut = N-1.
  return (N % 2 == 0) ? N : N - 1;
}

static void section(const char* title) {
  cout << "\n=== " << title << " ===\n" << flush;
}

int main() {
  // ---- Structural check: decode_maxcut has no invalid state. Any
  // bitstring is a valid 2-partition, so objective should NEVER be
  // negative -- unlike kcolor/clique, there is no "-1 = invalid" case to
  // watch for here. Confirm directly rather than assume from reading the
  // decoder. ----
  section("Structural: objective never negative (10 trials, N=10 random graph)");
  {
    Graph g = makeRandomGraph(10, 99);
    int negCount = 0;
    for (int seed = 0; seed < 10; seed++) {
      auto r = c2q_maxcut(g, 2, seed);
      if (r.objective < 0)
        negCount++;
    }
    cout << "  negative objectives: " << negCount << "/10\n" << flush;
  }

  // ---- Completeness: complete graph K_N, ground truth known
  // analytically (balanced bipartition). ----
  section("Completeness: complete graph K_N, layers=2, 10 seeds");
  for (int N : {4, 6, 8, 10, 12, 14, 16}) {
    Graph g = makeCompleteGraph(N);
    int truth = analyticalCompleteGraphCut(N);
    int pass = 0, total = 10;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_maxcut(g, 2, seed);
      if (r.objective == truth)
        pass++;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  N=" << N << "  truth=" << truth << "  pass=" << pass << "/" << total
         << "  (" << secs << "s for " << total << " seeds)\n" << flush;
  }

  // ---- Completeness: cycle graph C_N, ground truth known analytically. ----
  section("Completeness: cycle graph C_N, layers=2, 10 seeds");
  for (int N : {4, 6, 8, 10, 12, 14, 16}) {
    Graph g = makeCycleGraph(N);
    int truth = analyticalCycleCut(N);
    int pass = 0, total = 10;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_maxcut(g, 2, seed);
      if (r.objective == truth)
        pass++;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  N=" << N << "  truth=" << truth << "  pass=" << pass << "/" << total
         << "  (" << secs << "s for " << total << " seeds)\n" << flush;
  }

  // ---- Completeness: random sparse graph, ground truth via exhaustive
  // classical search (2^N enumeration, cheap up to N~20). This is the
  // "realistic" case -- irregular structure, not a symmetric graph a
  // balanced bipartition trivially solves. ----
  section("Completeness: random sparse graph (density 0.4), layers=2, 10 seeds");
  for (int N : {4, 6, 8, 10, 12, 14, 16}) {
    Graph g = makeRandomGraph(N, 42);
    int truth = classicalMaxCut(g);
    int pass = 0, total = 10;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_maxcut(g, 2, seed);
      if (r.objective == truth)
        pass++;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  N=" << N << "  truth=" << truth << "  pass=" << pass << "/" << total
         << "  (" << secs << "s for " << total << " seeds)\n" << flush;
  }

  // ---- Wall-clock cliff: single-seed spot checks past where 10-seed
  // batches stop being practical, on the complete-graph case. ----
  section("Wall-clock cliff: complete graph K_N, layers=2, single seed");
  for (int N : {18, 20, 22}) {
    Graph g = makeCompleteGraph(N);
    int truth = analyticalCompleteGraphCut(N);
    auto t0 = chrono::steady_clock::now();
    auto r = c2q_maxcut(g, 2, 0);
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  N=" << N << "  truth=" << truth << "  got=" << r.objective
         << "  " << (r.objective == truth ? "OK" : "MISS")
         << "  time=" << secs << "s\n" << flush;
  }

  // ---- Depth: does more QAOA depth rescue completeness on the harder
  // (random-graph) case at a fixed N, the way it did for clique? ----
  section("Depth sweep: random graph N=14, 5 seeds per layer count");
  {
    Graph g = makeRandomGraph(14, 42);
    int truth = classicalMaxCut(g);
    for (int layers : {2, 4, 6}) {
      int pass = 0, total = 5;
      auto t0 = chrono::steady_clock::now();
      for (int seed = 0; seed < total; seed++) {
        auto r = c2q_maxcut(g, layers, seed);
        if (r.objective == truth)
          pass++;
      }
      auto t1 = chrono::steady_clock::now();
      double secs = chrono::duration<double>(t1 - t0).count();
      cout << "  layers=" << layers << "  pass=" << pass << "/" << total
           << "  (" << secs << "s for " << total << " seeds)\n" << flush;
    }
  }

  cout << "\ndone\n";
  return 0;
}
