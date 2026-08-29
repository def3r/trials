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

static Graph makeStarGraph(int N) {
  // center=0 connected to all others, no other edges anywhere.
  // Max clique = 2 (center + any one leaf).
  Graph g;
  g.num_nodes = N;
  for (int i = 1; i < N; i++)
    g.edges.emplace_back(0, i, 1.0);
  return g;
}

static Graph makeEmptyGraph(int N) {
  Graph g;
  g.num_nodes = N;
  return g;
}

// Plant a clique of size `cliqueSize` on vertices [0, cliqueSize), then add
// sparse random noise edges among the rest (density ~0.15) so the target
// clique has to be found inside a bigger, mostly-unrelated graph.
static Graph makePlantedClique(int N, int cliqueSize, unsigned seed) {
  Graph g;
  g.num_nodes = N;
  for (int i = 0; i < cliqueSize; i++)
    for (int j = i + 1; j < cliqueSize; j++)
      g.edges.emplace_back(i, j, 1.0);
  mt19937 rng(seed);
  uniform_real_distribution<double> unif(0.0, 1.0);
  for (int i = 0; i < N; i++)
    for (int j = i + 1; j < N; j++) {
      if (i < cliqueSize && j < cliqueSize)
        continue;  // already added
      if (unif(rng) < 0.15)
        g.edges.emplace_back(i, j, 1.0);
    }
  return g;
}

static void section(const char* title) {
  cout << "\n=== " << title << " ===\n";
}

int main() {
  // ---- Soundness: infeasible target K, must NEVER report success ----
  section("Soundness: star graph (true max clique=2), request K=3");
  for (int N : {4, 6, 8, 10}) {
    Graph g = makeStarGraph(N);
    int falsepos = 0, total = 10;
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_clique(g, 3, 2, seed);
      if (r.objective >= 3)
        falsepos++;
    }
    cout << "  N=" << N << "  falsepos=" << falsepos << "/" << total << "\n";
  }

  section("Soundness: empty graph (no edges), request K=2");
  for (int N : {4, 8, 12}) {
    Graph g = makeEmptyGraph(N);
    int falsepos = 0, total = 10;
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_clique(g, 2, 2, seed);
      if (r.objective >= 2)
        falsepos++;
    }
    cout << "  N=" << N << "  falsepos=" << falsepos << "/" << total << "\n";
  }

  // ---- Completeness: trivially-feasible case (complete graph, any subset
  // is a valid clique) -- isolates whether QAOA can even hit the target
  // SIZE K, independent of clique validity ----
  section("Completeness: complete graph K_N, request K=N, layers=2");
  for (int N : {4, 6, 8, 10, 12, 14, 16}) {
    Graph g = makeCompleteGraph(N);
    int pass = 0, total = 10;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_clique(g, N, 2, seed);
      if (r.objective == N)
        pass++;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  N=" << N << "  pass=" << pass << "/" << total
         << "  (" << secs << "s for " << total << " seeds)\n";
  }

  // ---- Completeness: a small clique planted inside a larger, mostly
  // unrelated graph -- the realistic "needle in haystack" case ----
  section("Completeness: clique of size 4 planted in N-vertex sparse graph, request K=4, layers=2");
  for (int N : {6, 8, 10, 12, 14, 16}) {
    int pass = 0, total = 10;
    for (int seed = 0; seed < total; seed++) {
      Graph g = makePlantedClique(N, 4, /*graphSeed=*/seed);
      auto r = c2q_clique(g, 4, 2, seed);
      if (r.objective == 4)
        pass++;
    }
    cout << "  N=" << N << "  pass=" << pass << "/" << total << "\n";
  }

  // ---- Wall-clock cliff: single-seed spot checks beyond N=16, where
  // multi-seed statistics become impractically slow ----
  section("Wall-clock + reliability cliff: complete graph K_N, request K=N, single seed, layers=2");
  for (int N : {18, 20, 22}) {
    Graph g = makeCompleteGraph(N);
    auto t0 = chrono::steady_clock::now();
    auto r = c2q_clique(g, N, 2, /*seed=*/0);
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  N=" << N << "  qubits=" << N << "  objective=" << r.objective
         << "  time=" << secs << "s\n";
  }

  // ---- Does more QAOA depth help at a size where layers=2 already
  // struggles? ----
  section("Layers comparison at N=16 (complete graph, request K=16)");
  for (int layers : {2, 4, 6}) {
    Graph g = makeCompleteGraph(16);
    int pass = 0, total = 5;
    auto t0 = chrono::steady_clock::now();
    for (int seed = 0; seed < total; seed++) {
      auto r = c2q_clique(g, 16, layers, seed);
      if (r.objective == 16)
        pass++;
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    cout << "  layers=" << layers << "  pass=" << pass << "/" << total
         << "  (" << secs << "s for " << total << " seeds)\n";
  }

  // ---- Default k=-1 behaviour: what does it actually attempt? ----
  section("Default k=-1: what K does it target?");
  for (int N : {4, 6, 8}) {
    Graph g = makeCompleteGraph(N);
    auto r = c2q_clique(g, /*k=*/-1);
    cout << "  N=" << N << "  (expect K=N-1=" << (N - 1)
         << ")  objective=" << r.objective << "\n";
  }

  return 0;
}
