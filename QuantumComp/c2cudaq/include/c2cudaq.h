#pragma once
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Graph input type
// All graph problems share this representation.
// Vertices are 0-indexed integers.  For unweighted graphs set weight = 1.0.
struct Graph {
  int num_nodes;
  std::vector<std::tuple<int, int, double>> edges;  // (u, v, weight)
};

// Result types
struct GraphResult {
  // Raw QAOA/VQE output bitstring (length = num_qubits used by the problem).
  // For KColor: num_nodes*k bits (one-hot per node).
  // For TSP:    num_nodes^2 bits (one-hot per city-position).
  // For all others: num_nodes bits (one bit per node).
  std::string partition;
  // Classical objective value decoded from partition.
  // MaxCut  : total weight of cut edges  (−1 if partition is invalid)
  // MIS     : size of independent set    (−1 if set is not independent)
  // VC      : size of vertex cover       (−1 if cover is incomplete)
  // Clique  : size of clique             (−1 if not a clique)
  // KColor  : 0 if valid, −1 if invalid
  // TSP     : total tour distance        (−1.0 cast to int if invalid)
  int objective;
  double energy;  // QAOA/VQE expectation value at optimum
};

// Arithmetic
// Quantum ripple-carry adder.  Result register has n_bits+1 qubits.
// n_bits auto-computed as max(bit_length(a), bit_length(b)) + 1.
// Safe range: |a|, |b| ≤ 255  (n_bits ≤ 8 → 25 total qubits).
int64_t c2q_add(int64_t a, int64_t b);

// Quantum subtraction via two's-complement add.
// Same qubit limits as c2q_add.  Returns signed result.
int64_t c2q_sub(int64_t a, int64_t b);

// QFT-based quantum multiplier (symmetric operands).
// Safe range: a, b ≤ 15  (n_qubits = 4 → 12 total qubits).
int64_t c2q_mul(int64_t a, int64_t b);

// QFT multiplier with per-operand bit widths.
// Safe range: size_a + size_b ≤ 14 qubits total in accumulator.
int64_t c2q_mul(int64_t a, int64_t b, int size_a, int size_b);

// Factorization
// Finds a non-trivial factor pair (p, q) with p*q == n using quantum
// multiplication for verification.  Returns {1, n} if no factor is found. Note:
// uses QFT multiplier, not full Grover; simulator-only. Safe range: n ≤ 255
// (result register ≤ 8 qubits).
std::pair<int64_t, int64_t> c2q_factor(int64_t n);

// SQIF (Sublinear Quantum Integer Factorization): classical LLL lattice
// reduction + Babai's algorithm sets up a closest-vector problem, QAOA
// refines it, and classical postprocessing (smoothness test + GF(2) linear
// algebra over collected sr-pairs) extracts the factors. Qubit count is
// sublinear in bit-length of n (paper: arXiv:2212.12372), so this is a
// separate, independent path from c2q_factor -- not a replacement, not
// dispatched automatically. Currently validated against the paper's own
// three worked examples (n = 1961, 48567227, 261980999226229); other n
// fall back to the paper's approximate dimension formula (lower
// confidence -- see claude.md). Returns {1, n} if no factor is found.
std::pair<int64_t, int64_t> c2q_factor_sqif(int64_t n);

// Graph problems - QAOA
// All graph functions run QAOA with COBYLA optimizer.
// layers: QAOA circuit depth p (higher → better quality, slower).
// seed:   random seed for initial parameters.
GraphResult c2q_maxcut(const Graph& g, int layers = 2, int seed = 13);
GraphResult c2q_mis(const Graph& g, int layers = 2, int seed = 13);
GraphResult c2q_vc(const Graph& g, int layers = 2, int seed = 13);
GraphResult c2q_clique(const Graph& g,
                       int k = -1,
                       int layers = 2,
                       int seed = 13);
GraphResult c2q_kcolor(const Graph& g,
                       int k = 3,
                       int layers = 2,
                       int seed = 13);
GraphResult c2q_tsp(const Graph& g, int layers = 2, int seed = 13);

// Graph problems - VQE
// Hardware-efficient RY+CZ ansatz.  reps: number of entanglement layers.
GraphResult c2q_maxcut_vqe(const Graph& g, int reps = 2, int seed = 13);
GraphResult c2q_mis_vqe(const Graph& g, int reps = 2, int seed = 13);
GraphResult c2q_vc_vqe(const Graph& g, int reps = 2, int seed = 13);
GraphResult c2q_clique_vqe(const Graph& g,
                           int k = -1,
                           int reps = 2,
                           int seed = 13);
GraphResult c2q_kcolor_vqe(const Graph& g,
                           int k = 3,
                           int reps = 2,
                           int seed = 13);
GraphResult c2q_tsp_vqe(const Graph& g, int reps = 2, int seed = 13);
