#pragma once
#include <c2cudaq.h>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq {
class sample_result;
}

namespace c2cudaq {

// Qubit safety check
// Statevector requires 2^n * 16 bytes.  28 qubits ≈ 4 GB.
inline void check_qubit_limit(int n, const char* who) {
  if (n > 28)
    throw std::runtime_error(std::string(who) + " needs " + std::to_string(n) +
                             " qubits (limit 28). Reduce problem size.");
}

// Ising representation
// Ising Hamiltonian: H = Σ J_ij Z_i Z_j  +  Σ h_i Z_i  +  offset
struct IsingTerms {
  std::vector<int> zz_i, zz_j;  // ZZ interaction indices
  std::vector<double> zz_c;     // ZZ coefficients J_ij
  std::vector<int> z_i;         // Z-field indices
  std::vector<double> z_c;      // Z-field coefficients h_i
  double offset = 0.0;
};

// QAOA optimization loop (defined in qaoa.cpp). Runs COBYLA over a
// p-layer qaoa_general circuit for the given Ising Hamiltonian and returns
// the top sampled bitstring, its energy, and the optimized (gamma, beta)
// parameters. The parameters are returned so callers needing more than one
// candidate bitstring (e.g. sqif_qaoa.cpp) can re-run cudaq::sample
// themselves at the optimum instead of re-optimizing.
std::tuple<std::string, double, std::vector<double>>
run_qaoa(int n_qubits, int layers, const IsingTerms& ising, int seed);

// Full sampled histogram at a given (already-optimized) parameter set --
// see run_qaoa above for why this is separate. Declared here as returning
// cudaq::sample_result (forward-declared below) so this header stays
// includable from pure-host translation units without pulling in cudaq.h;
// callers that use the return value (sqif.cpp) include cudaq.h themselves.
cudaq::sample_result sqif_sample_qaoa(int n_qubits, int layers,
                                       const IsingTerms& ising,
                                       const std::vector<double>& params);

// QUBO → Ising conversion
// Q is an n×n upper-triangular QUBO matrix (stored row-major, size n*n).
// x_i = (1 - sigma_i) / 2  =>  H_ising = qubo_to_ising(Q) + offset constant.
inline IsingTerms qubo_to_ising(const std::vector<double>& Q, int n) {
  IsingTerms ising;
  for (int i = 0; i < n; ++i) {
    // Diagonal → Z-field
    double off_sum = 0.0;
    for (int j = i + 1; j < n; ++j)
      off_sum += Q[i * n + j];
    for (int j = 0; j < i; ++j)
      off_sum += Q[j * n + i];
    double h = -0.5 * Q[i * n + i] - 0.25 * off_sum;
    if (std::abs(h) > 1e-12) {
      ising.z_i.push_back(i);
      ising.z_c.push_back(h);
    }
    ising.offset += 0.5 * Q[i * n + i];

    // Off-diagonal → ZZ
    for (int j = i + 1; j < n; ++j) {
      double J = 0.25 * Q[i * n + j];
      if (std::abs(J) > 1e-12) {
        ising.zz_i.push_back(i);
        ising.zz_j.push_back(j);
        ising.zz_c.push_back(J);
      }
      ising.offset += 0.25 * Q[i * n + j];
    }
  }
  return ising;
}

// QUBO builders
// All return an n×n matrix stored row-major in a flat vector.

inline std::vector<double> qubo_maxcut(const Graph& g) {
  int n = g.num_nodes;
  std::vector<double> Q(n * n, 0.0);
  for (auto& [u, v, w] : g.edges) {
    int i = std::min(u, v), j = std::max(u, v);
    Q[i * n + i] -= w;
    Q[j * n + j] -= w;
    Q[i * n + j] += 2.0 * w;
  }
  return Q;
}

inline std::vector<double> qubo_mis(const Graph& g, double B = 1.0) {
  int n = g.num_nodes;
  std::vector<double> Q(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    Q[i * n + i] -= B;
  for (auto& [u, v, w] : g.edges) {
    int i = std::min(u, v), j = std::max(u, v);
    Q[i * n + j] += 2.0 * B;
  }
  return Q;
}

inline std::vector<double> qubo_vc(const Graph& g, double B = 1.0) {
  int n = g.num_nodes;
  std::vector<double> Q(n * n, 0.0);
  for (int i = 0; i < n; ++i)
    Q[i * n + i] += B;
  for (auto& [u, v, w] : g.edges) {
    int i = std::min(u, v), j = std::max(u, v);
    Q[i * n + i] -= 2.0 * B;
    Q[j * n + j] -= 2.0 * B;
    Q[i * n + j] += 2.0 * B;
  }
  return Q;
}

inline std::vector<double> qubo_clique(const Graph& g,
                                       int K = -1,
                                       double B = 1.0) {
  int n = g.num_nodes;
  if (K < 0)
    K = std::max(1, n - 1);
  K = std::max(1, std::min(K, n));
  double A = K * B + 10.0;
  std::vector<double> Q(n * n, 0.0);
  double lc = -2.0 * A * K + A;
  for (int i = 0; i < n; ++i)
    Q[i * n + i] += lc;
  std::set<std::pair<int, int>> edge_set;
  for (auto& [u, v, w] : g.edges) {
    edge_set.insert({u, v});
    edge_set.insert({v, u});
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j) {
      Q[i * n + j] += 2.0 * A;
      if (edge_set.count({i, j}))
        Q[i * n + j] -= B;
    }
  return Q;
}

inline std::vector<double> qubo_kcolor(const Graph& g,
                                       int k = 3,
                                       double A = 2.0) {
  int N = g.num_nodes, dim = N * k;
  std::vector<double> Q(dim * dim, 0.0);
  // One-color-per-vertex constraint
  for (int v = 0; v < N; ++v) {
    for (int c = 0; c < k; ++c)
      Q[(v * k + c) * dim + (v * k + c)] -= A;
    for (int c = 0; c < k; ++c)
      for (int d = c + 1; d < k; ++d)
        Q[(v * k + c) * dim + (v * k + d)] += 2.0 * A;
  }
  // Adjacent vertices must have different colors
  for (auto& [u, v, w] : g.edges)
    for (int c = 0; c < k; ++c) {
      int i = std::min(u, v) * k + c, j = std::max(u, v) * k + c;
      Q[i * dim + j] += A;
    }
  return Q;
}

inline std::vector<double> qubo_tsp(const Graph& g, double B = 1.0) {
  int n = g.num_nodes, dim = n * n;
  double max_w = 1.0;
  for (auto& [u, v, w] : g.edges)
    max_w = std::max(max_w, w);
  double A = B * max_w + 10.0;
  std::vector<double> Q(dim * dim, 0.0);
  // Build weight lookup
  std::map<std::pair<int, int>, double> wmap;
  for (auto& [u, v, w] : g.edges) {
    wmap[{u, v}] = w;
    wmap[{v, u}] = w;
  }
  // Distance cost H_B
  //
  // Two bugs fixed here, both confirmed empirically (see
  // loopanalysis/analysis/tsp/tsp_qubo_check.cpp and analysis/tsp/tsp.md
  // in the LLVM-passes project that found this):
  //
  // 1. Wraparound: the loop used to run `p` over [0, n-2], coupling only
  //    position p to position p+1 -- it never coupled position n-1 back
  //    to position 0, so the tour's closing edge carried zero cost. But
  //    decode_tsp (the function that SCORES a returned tour) does close
  //    the loop, via `tour[(p+1) % n]`. That mismatch meant the QUBO's
  //    own true minimum wasn't always the actual TSP optimum -- verified
  //    directly at N=4: the old QUBO's minimum was a valid tour of
  //    length 48, while the true optimum was 35. Now `p` runs over the
  //    full [0, n-1] with the next position taken mod n, matching
  //    decode_tsp exactly (and matching the standard Lucas 2014 TSP QUBO
  //    formulation, which is defined for a closed Hamiltonian cycle).
  //
  // 2. Lower-triangle writes silently dropped: Q is documented (and
  //    consumed by qubo_to_ising, above) as upper-triangular only --
  //    every read in qubo_to_ising accesses Q[a*n+b] with a<b. The old
  //    write `Q[(v*n+p)*dim + (u*n+p+1)]` is NOT canonically ordered:
  //    whenever v>u, algebraically (v*n+p) > (u*n+p+1) for any valid p<n
  //    (the n-multiplier dominates the +1 offset), so the write lands in
  //    the LOWER triangle -- a cell qubo_to_ising never reads. Since the
  //    outer loop visits every ordered (v,u) pair with an edge, exactly
  //    half of all directed distance-cost terms (every v>u pair) were
  //    silently discarded, not merely misweighted. Confirmed directly on
  //    a 4-node complete graph: 18 nonzero entries landed in the lower
  //    triangle, matching the predicted count (6 v>u pairs x 3
  //    positions) exactly. Fixed by always writing to the canonical
  //    (min, max) cell.
  for (int v = 0; v < n; ++v)
    for (int u = 0; u < n; ++u) {
      if (u == v)
        continue;
      auto it = wmap.find({v, u});
      if (it == wmap.end())
        continue;
      double w = it->second;
      for (int p = 0; p < n; ++p) {
        int p2 = (p + 1) % n;
        int i = v * n + p, j = u * n + p2;
        if (i == j)
          continue;
        int lo = std::min(i, j), hi = std::max(i, j);
        Q[lo * dim + hi] += B * w;
      }
    }
  // Each city visits exactly one position
  for (int v = 0; v < n; ++v) {
    for (int j = 0; j < n; ++j) {
      Q[(v * n + j) * dim + (v * n + j)] -= A;
      for (int k2 = j + 1; k2 < n; ++k2)
        Q[(v * n + j) * dim + (v * n + k2)] += 2.0 * A;
    }
  }
  // Each position has exactly one city
  for (int j = 0; j < n; ++j) {
    for (int v = 0; v < n; ++v) {
      Q[(v * n + j) * dim + (v * n + j)] -= A;
      for (int u = v + 1; u < n; ++u)
        Q[(v * n + j) * dim + (u * n + j)] += 2.0 * A;
    }
  }
  // Non-edges penalty -- same two fixes as H_B above: wraparound
  // included, and writes canonically ordered so they aren't silently
  // dropped by qubo_to_ising's upper-triangle-only reads.
  for (int v = 0; v < n; ++v)
    for (int u = 0; u < n; ++u) {
      if (u == v)
        continue;
      if (wmap.count({v, u}))
        continue;
      for (int p = 0; p < n; ++p) {
        int p2 = (p + 1) % n;
        int i = v * n + p, j = u * n + p2;
        if (i == j)
          continue;
        int lo = std::min(i, j), hi = std::max(i, j);
        Q[lo * dim + hi] += A;
      }
    }
  return Q;
}

// Result decoders
// decode_partition: bitstring → indices of nodes in the '1' partition.
// Used by MaxCut, MIS, VC, and Clique (all use one bit per node).
inline std::vector<int> decode_partition(const std::string& bits) {
  std::vector<int> nodes;
  for (int i = 0; i < (int)bits.size(); ++i)
    if (bits[i] == '1')
      nodes.push_back(i);
  return nodes;
}

// decode_kcolor_assignment: bitstring → color index per node (-1 if unset).
// bits has N*k chars; node v has color c if bits[v*k + c] == '1'.
inline std::vector<int> decode_kcolor_assignment(const std::string& bits,
                                                  int N, int k) {
  std::vector<int> color(N, -1);
  for (int v = 0; v < N; ++v)
    for (int c = 0; c < k; ++c)
      if ((int)bits.size() > v * k + c && bits[v * k + c] == '1') {
        color[v] = c;
        break;
      }
  return color;
}

// decode_tsp_tour: bitstring → city visit order (-1 positions if invalid).
// bits has N^2 chars; city v is at position p if bits[v*N + p] == '1'.
inline std::vector<int> decode_tsp_tour(const std::string& bits, int N) {
  std::vector<int> tour(N, -1);
  for (int v = 0; v < N; ++v)
    for (int p = 0; p < N; ++p)
      if ((int)bits.size() > v * N + p && bits[v * N + p] == '1') {
        tour[p] = v;
        break;
      }
  return tour;
}

// Objective decoders
inline int decode_maxcut(const std::string& bits, const Graph& g) {
  int cut = 0;
  for (auto& [u, v, w] : g.edges)
    if (bits[u] != bits[v])
      cut += (int)w;
  return cut;
}

inline int decode_mis(const std::string& bits, const Graph& g) {
  for (auto& [u, v, w] : g.edges)
    if (bits[u] == '1' && bits[v] == '1')
      return -1;
  int sz = 0;
  for (char b : bits)
    if (b == '1')
      ++sz;
  return sz;
}

inline int decode_vc(const std::string& bits, const Graph& g) {
  for (auto& [u, v, w] : g.edges)
    if (bits[u] == '0' && bits[v] == '0')
      return -1;
  int sz = 0;
  for (char b : bits)
    if (b == '1')
      ++sz;
  return sz;
}

inline int decode_clique(const std::string& bits, const Graph& g) {
  std::set<std::pair<int, int>> edge_set;
  for (auto& [u, v, w] : g.edges) {
    edge_set.insert({u, v});
    edge_set.insert({v, u});
  }
  std::vector<int> nodes;
  for (int i = 0; i < (int)bits.size(); ++i)
    if (bits[i] == '1')
      nodes.push_back(i);
  for (int i = 0; i < (int)nodes.size(); ++i)
    for (int j = i + 1; j < (int)nodes.size(); ++j)
      if (!edge_set.count({nodes[i], nodes[j]}))
        return -1;
  return (int)nodes.size();
}

// bits has N*k chars; returns 0 if valid coloring, -1 otherwise.
inline int decode_kcolor(const std::string& bits, const Graph& g, int k) {
  int N = g.num_nodes;
  std::vector<int> color(N, -1);
  for (int v = 0; v < N; ++v)
    for (int c = 0; c < k; ++c)
      if ((int)bits.size() > v * k + c && bits[v * k + c] == '1') {
        color[v] = c;
        break;
      }
  for (auto& [u, v, w] : g.edges)
    if (color[u] < 0 || color[v] < 0 || color[u] == color[v])
      return -1;
  return 0;
}

// bits has N^2 chars; returns tour distance or -1 if invalid.
inline int decode_tsp(const std::string& bits, const Graph& g) {
  int n = g.num_nodes;
  std::map<std::pair<int, int>, double> wmap;
  for (auto& [u, v, w] : g.edges) {
    wmap[{u, v}] = w;
    wmap[{v, u}] = w;
  }
  std::vector<int> tour(n, -1);
  for (int v = 0; v < n; ++v)
    for (int p = 0; p < n; ++p)
      if ((int)bits.size() > v * n + p && bits[v * n + p] == '1') {
        tour[p] = v;
        break;
      }
  std::set<int> visited(tour.begin(), tour.end());
  if ((int)visited.size() != n || visited.count(-1))
    return -1;
  double dist = 0.0;
  for (int p = 0; p < n; ++p) {
    auto it = wmap.find({tour[p], tour[(p + 1) % n]});
    if (it == wmap.end())
      return -1;
    dist += it->second;
  }
  return (int)dist;
}

}  // namespace c2cudaq
