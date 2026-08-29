#pragma once
#include <string>
#include <utility>
#include <vector>

struct MaxCutResult {
  std::string partition;  // bitstring over vertices, e.g. "01011"
  int cut_value;          // number of edges crossing the partition
  double optimal_energy;  // QAOA expectation at optimum (~-cut_value)
};

// Solve max-cut via QAOA simulation.
// edges: list of (u, v) undirected edge pairs, 0-indexed vertices
// layer_count: QAOA depth p (default 2, higher = better quality but slower)
MaxCutResult solve_maxcut(int num_nodes,
                          const std::vector<std::pair<int, int>>& edges,
                          int layer_count = 2);
