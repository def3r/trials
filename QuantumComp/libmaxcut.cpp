#include "libmaxcut.h"
#include <cudaq.h>
#include <cudaq/optimizers.h>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

// QAOA problem unitary for one edge: CNOT – RZ(2α) – CNOT
struct qaoa_edge {
  __qpu__ void operator()(cudaq::qubit& q0, cudaq::qubit& q1, double alpha) {
    x<cudaq::ctrl>(q0, q1);
    rz(2.0 * alpha, q1);
    x<cudaq::ctrl>(q0, q1);
  }
};

// Full QAOA circuit: p layers of (problem unitary + mixer)
struct qaoa_circuit {
  __qpu__ void operator()(int n,
                          int p,
                          std::vector<int> src,
                          std::vector<int> tgt,
                          std::vector<double> thetas) {
    cudaq::qvector qreg(n);
    h(qreg);

    int n_edges = src.size();
    for (int l = 0; l < p; ++l) {
      for (int e = 0; e < n_edges; ++e)
        qaoa_edge{}(qreg[src[e]], qreg[tgt[e]], thetas[l]);
      for (int j = 0; j < n; ++j)
        rx(2.0 * thetas[l + p], qreg[j]);
    }
  }
};

// H = Σ_{(u,v)∈E} 0.5*(Z_u Z_v - I_u I_v)  — same as Python hamiltonian_max_cut
static cudaq::spin_op build_hamiltonian(const std::vector<int>& src,
                                        const std::vector<int>& tgt) {
  auto H = 0.5 * (cudaq::spin::z(src[0]) * cudaq::spin::z(tgt[0]) -
                  cudaq::spin::i(src[0]) * cudaq::spin::i(tgt[0]));
  for (std::size_t e = 1; e < src.size(); ++e)
    H += 0.5 * (cudaq::spin::z(src[e]) * cudaq::spin::z(tgt[e]) -
                cudaq::spin::i(src[e]) * cudaq::spin::i(tgt[e]));
  return H;
}

static int compute_cut(const std::string& partition,
                       const std::vector<std::pair<int, int>>& edges) {
  int cut = 0;
  for (auto [u, v] : edges)
    if (partition[u] != partition[v])
      ++cut;
  return cut;
}

MaxCutResult solve_maxcut(int num_nodes,
                          const std::vector<std::pair<int, int>>& edges,
                          int layer_count) {
  std::vector<int> src, tgt;
  for (auto [u, v] : edges) {
    src.push_back(u);
    tgt.push_back(v);
  }

  int param_count = 2 * layer_count;
  auto hamiltonian = build_hamiltonian(src, tgt);

  // Random init in [-π/8, π/8], seeded to match Python (seed 13)
  std::mt19937 rng(13);
  std::uniform_real_distribution<double> dist(-M_PI / 8.0, M_PI / 8.0);
  std::vector<double> init_params(param_count);
  for (auto& p : init_params)
    p = dist(rng);

  cudaq::optimizers::cobyla optimizer;
  optimizer.initial_parameters = init_params;

  auto [opt_val, opt_params] = optimizer.optimize(
      param_count,
      [&](std::vector<double> params, std::vector<double>& /*grad*/) -> double {
        return cudaq::observe(qaoa_circuit{}, hamiltonian, num_nodes,
                              layer_count, src, tgt, params)
            .expectation();
      });

  auto counts = cudaq::sample(qaoa_circuit{}, num_nodes, layer_count, src, tgt,
                              opt_params);
  auto partition = counts.most_probable();

  return {partition, compute_cut(partition, edges), opt_val};
}
