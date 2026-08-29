// Pure-classical verification of qubo_tsp/decode_tsp -- no cudaq, no GPU.
// Question: does the TSP QUBO's own minimum (found by brute force over all
// bitstrings) actually correspond to a valid, optimal tour? If yes, the
// stress test's observed collapse is a QAOA-convergence problem, not a QUBO
// construction bug. If no, something in qubo_tsp is genuinely wrong.
#include <c2cudaq.h>
#include <c2cudaq/internal.h>
#include <algorithm>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
using namespace std;
using namespace c2cudaq;

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

// Raw QUBO objective x^T Q x (Q stored row-major, dim x dim -- upper
// triangle only is populated by qubo_tsp, so this sums both x_i*Q_ij*x_j
// for i<j as well as the diagonal, matching how qubo_to_ising would
// interpret it).
static double quboObjective(const vector<double>& Q, int dim, unsigned mask) {
  double val = 0.0;
  for (int i = 0; i < dim; i++) {
    if (!((mask >> i) & 1)) continue;
    val += Q[i * dim + i];
    for (int j = i + 1; j < dim; j++)
      if ((mask >> j) & 1)
        val += Q[i * dim + j];
  }
  return val;
}

static string bitsFromMask(unsigned mask, int dim) {
  string s(dim, '0');
  for (int i = 0; i < dim; i++)
    if ((mask >> i) & 1) s[i] = '1';
  return s;
}

int main() {
  for (int N : {3, 4}) {
    unsigned seed = (N == 3) ? 1 : 2;  // matches tsp_stress.cpp exactly
    Graph g = makeCompleteWeightedGraph(N, seed);
    int dim = N * N;
    auto Q = qubo_tsp(g);
    int truth = classicalTsp(g);

    cout << "=== N=" << N << " (dim=" << dim << " qubits), classical truth=" << truth << " ===\n";

    // Brute force over all 2^dim bitstrings: find the QUBO objective's own
    // minimum, and separately track the best objective among VALID
    // permutation bitstrings only.
    double bestAny = 1e18; unsigned bestAnyMask = 0;
    double bestValid = 1e18; unsigned bestValidMask = 0;
    long validCount = 0;

    for (unsigned mask = 0; mask < (1u << dim); mask++) {
      double v = quboObjective(Q, dim, mask);
      if (v < bestAny) { bestAny = v; bestAnyMask = mask; }
      string bits = bitsFromMask(mask, dim);
      int obj = decode_tsp(bits, g);
      if (obj >= 0) {
        validCount++;
        if (v < bestValid) { bestValid = v; bestValidMask = mask; }
      }
    }

    string anyBits = bitsFromMask(bestAnyMask, dim);
    int anyObj = decode_tsp(anyBits, g);
    string validBits = bitsFromMask(bestValidMask, dim);
    int validObj = decode_tsp(validBits, g);

    cout << "  valid permutation bitstrings: " << validCount << " / " << (1u << dim) << "\n";
    cout << "  global QUBO minimum: objective=" << quboObjective(Q, dim, bestAnyMask)
         << "  bits=" << anyBits
         << "  decode_tsp=" << anyObj
         << (anyObj >= 0 ? (anyObj == truth ? "  [VALID, OPTIMAL]" : "  [VALID, SUBOPTIMAL]")
                          : "  [INVALID -- QUBO minimum is NOT a valid tour!]")
         << "\n";
    cout << "  best-among-valid: objective=" << bestValid
         << "  bits=" << validBits
         << "  decode_tsp=" << validObj
         << (validObj == truth ? "  [OPTIMAL]" : "  [SUBOPTIMAL]")
         << "\n";

    if (anyObj < 0) {
      cout << "  ==> BUG CONFIRMED: the QUBO's own unconstrained minimum is an invalid\n"
           << "      tour, not the optimal one -- QAOA is being asked to converge on\n"
           << "      a target that doesn't correspond to a real solution.\n";
    } else if (anyObj != truth) {
      cout << "  ==> BUG: QUBO minimum is a valid tour but not the OPTIMAL one\n"
           << "      (classical truth=" << truth << ", QUBO's favorite=" << anyObj << ").\n";
    } else {
      cout << "  ==> QUBO construction is correct: its unconstrained minimum IS the\n"
           << "      true optimal tour. Any collapse in the stress test is a QAOA\n"
           << "      convergence problem, not a QUBO/decoder bug.\n";
    }

    // Explicitly test the "missing wraparound edge" hypothesis: take the
    // best valid tour, then check whether swapping which edge closes the
    // loop (keeping the same set of visited pairs at every OTHER position)
    // changes the QUBO objective at all. If it doesn't, H_B has no cost
    // term for the wraparound edge.
    {
      vector<int> tour = decode_tsp_tour(validBits, N);
      // Build two bitstrings: original tour, and a rotated tour that
      // changes only which two cities occupy positions 0 and N-1 (keeps
      // every other adjacency the same, but changes the closing edge).
      // Simplest such transform: reverse the whole tour. Reversing keeps
      // ALL consecutive-pair edges the same set (edges are symmetric,
      // undirected), including the wraparound -- not useful. Instead,
      // rotate by one position: new position p holds tour[(p+1)%N]. This
      // changes the closing edge from (tour[N-1],tour[0]) to
      // (tour[0],tour[1]) while every OTHER consecutive pair in the
      // interior shifts identity but stays a used edge in the same cyclic
      // order -- not a clean isolation either for general N.
      // Cleanest isolation: for N=4 specifically, compare two tours that
      // agree on positions 0,1,2 and only differ in the pairing that
      // closes position 3 back to position 0, by swapping which of the
      // two remaining cities goes last -- only possible cleanly by
      // picking two tours that are identical except for a single
      // transposition of the last two cities, which also changes one
      // interior edge. Given the complexity of isolating just the
      // wraparound edge in general, report it qualitatively instead: sum
      // the H_B coefficient actually present at the wraparound offset.
      double wraparoundCoeff = 0.0;
      bool anyWraparoundTerm = false;
      for (int v = 0; v < N; v++)
        for (int u = 0; u < N; u++) {
          if (u == v) continue;
          int i = v * N + (N - 1);  // (v, position N-1)
          int j = u * N + 0;        // (u, position 0)
          int lo = min(i, j), hi = max(i, j);
          double c = Q[lo * dim + hi];
          // A "true" H_B distance term here would be a small positive
          // value near the edge weight (up to 20); the constraint terms
          // (2*A, A ~ 10-30+) are much larger and appear at DIFFERENT
          // (i,j) pairs (same v or same p only) -- (v,N-1)-(u,0) with
          // v!=u is never touched by the "one city per position"/"one
          // position per city" loops (those only pair same-v or same-p
          // indices), so any nonzero value here must come from H_B or the
          // non-edges penalty.
          if (c != 0.0) { anyWraparoundTerm = true; wraparoundCoeff += fabs(c); }
        }
      cout << "  wraparound (position N-1 -> position 0) QUBO coupling present: "
           << (anyWraparoundTerm ? "yes" : "NO -- confirms H_B has no wraparound cost term")
           << "\n";
    }
    cout << "\n";
  }
  return 0;
}
