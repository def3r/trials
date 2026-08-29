#include <c2cudaq.h>
#include <c2cudaq/internal.h>
#include <cudaq.h>
#include <cudaq/optimizers.h>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace c2cudaq;

// Generalized QAOA kernel
// Cost layer: ZZ terms → CNOT+RZ+CNOT; Z terms → RZ.
// Mixer:      RX on all qubits.
// thetas: [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]
struct qaoa_general {
    __qpu__ void operator()(int n, int p,
                            std::vector<int>    zz_i, std::vector<int>    zz_j,
                            std::vector<double> zz_c,
                            std::vector<int>    z_i,  std::vector<double> z_c,
                            std::vector<double> thetas) {
        cudaq::qvector qreg(n);
        h(qreg);

        int nzz = zz_i.size(), nz = z_i.size();
        for (int l = 0; l < p; ++l) {
            double gamma = thetas[l], beta = thetas[l + p];
            for (int e = 0; e < nzz; ++e) {
                x<cudaq::ctrl>(qreg[zz_i[e]], qreg[zz_j[e]]);
                rz(2.0 * gamma * zz_c[e], qreg[zz_j[e]]);
                x<cudaq::ctrl>(qreg[zz_i[e]], qreg[zz_j[e]]);
            }
            for (int e = 0; e < nz; ++e)
                rz(2.0 * gamma * z_c[e], qreg[z_i[e]]);
            for (int j = 0; j < n; ++j)
                rx(2.0 * beta, qreg[j]);
        }
    }
};

// Hardware-efficient VQE ansatz (RY + linear CZ)
// thetas: n*(reps+1) angles - one RY per qubit per layer.
struct vqe_ansatz {
    __qpu__ void operator()(int n, int reps, std::vector<double> thetas) {
        cudaq::qvector qreg(n);
        for (int j = 0; j < n; ++j) ry(thetas[j], qreg[j]);
        for (int r = 0; r < reps; ++r) {
            for (int j = 0; j < n - 1; ++j)
                z<cudaq::ctrl>(qreg[j], qreg[j + 1]);
            int off = n * (r + 1);
            for (int j = 0; j < n; ++j) ry(thetas[off + j], qreg[j]);
        }
    }
};

// Build cudaq::spin_op from IsingTerms
static cudaq::spin_op make_hamiltonian(const IsingTerms& ising) {
    cudaq::spin_op H;
    bool first = true;
    auto add = [&](cudaq::spin_op t) {
        if (first) { H = t; first = false; } else H += t;
    };
    for (std::size_t e = 0; e < ising.zz_i.size(); ++e)
        add(ising.zz_c[e] * cudaq::spin::z(ising.zz_i[e]) *
                             cudaq::spin::z(ising.zz_j[e]));
    for (std::size_t e = 0; e < ising.z_i.size(); ++e)
        add(ising.z_c[e] * cudaq::spin::z(ising.z_i[e]));
    if (first) add(0.0 * cudaq::spin::i(0)); // zero Hamiltonian guard
    return H;
}

namespace c2cudaq {

// Full sampled histogram (not just most_probable) at a given parameter
// set. Used by the SQIF path, which needs many candidate bitstrings per
// circuit (Stage 3 sr-pair collection), unlike the graph-problem callers
// of run_qaoa below which only need the single best partition.
cudaq::sample_result sqif_sample_qaoa(int n_qubits, int layers,
                                       const IsingTerms& ising,
                                       const std::vector<double>& params) {
    return cudaq::sample(qaoa_general{}, n_qubits, layers,
                          ising.zz_i, ising.zz_j, ising.zz_c,
                          ising.z_i, ising.z_c, params);
}

// QAOA optimization loop
// Exposed (non-static, in namespace c2cudaq to match internal.h's
// declaration) so sqif_qaoa.cpp can reuse it for the SQIF path instead of
// duplicating a second QAOA optimizer loop. Returns opt_par too (not just
// the top bitstring) so callers that need more than the single
// most-probable state (SQIF samples many candidate bitstrings per circuit)
// can re-run cudaq::sample themselves.
std::tuple<std::string, double, std::vector<double>>
run_qaoa(int n_qubits, int layers, const IsingTerms& ising, int seed) {
    auto H    = make_hamiltonian(ising);
    int  npar = 2 * layers;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-M_PI / 8.0, M_PI / 8.0);
    std::vector<double> init(npar);
    for (auto& v : init) v = dist(rng);

    cudaq::optimizers::cobyla opt;
    opt.initial_parameters = init;
    // Stop well before NLOpt collapses the simplex to floating-point noise.
    // Quantum shot noise is ~1e-2, so 1e-3 tolerance is already over-converged.
    opt.f_tol    = 1e-10;
    opt.max_eval = 500;

    double best_val = std::numeric_limits<double>::max();
    std::vector<double> best_par = init;

    double opt_val; std::vector<double> opt_par;
    try {
        auto [v, p] = opt.optimize(
            npar,
            [&](std::vector<double> par, std::vector<double>&) -> double {
                double e = cudaq::observe(qaoa_general{}, H,
                    n_qubits, layers,
                    ising.zz_i, ising.zz_j, ising.zz_c,
                    ising.z_i,  ising.z_c,  par).expectation();
                if (e < best_val) { best_val = e; best_par = par; }
                return e;
            });
        opt_val = v; opt_par = p;
    } catch (...) {
        opt_val = best_val; opt_par = best_par;
    }

    auto counts = cudaq::sample(qaoa_general{},
        n_qubits, layers,
        ising.zz_i, ising.zz_j, ising.zz_c,
        ising.z_i,  ising.z_c,  opt_par);
    return {counts.most_probable(), opt_val, opt_par};
}
}  // namespace c2cudaq

// VQE optimization loop
static std::pair<std::string, double>
run_vqe(int n_qubits, int reps, const IsingTerms& ising, int seed) {
    auto H    = make_hamiltonian(ising);
    int  npar = n_qubits * (reps + 1);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-M_PI / 4.0, M_PI / 4.0);
    std::vector<double> init(npar);
    for (auto& v : init) v = dist(rng);

    cudaq::optimizers::cobyla opt;
    opt.initial_parameters = init;
    opt.f_tol    = 1e-3;
    opt.max_eval = 500;

    double best_val = std::numeric_limits<double>::max();
    std::vector<double> best_par = init;

    double opt_val; std::vector<double> opt_par;
    try {
        auto [v, p] = opt.optimize(
            npar,
            [&](std::vector<double> par, std::vector<double>&) -> double {
                double e = cudaq::observe(vqe_ansatz{}, H, n_qubits, reps, par).expectation();
                if (e < best_val) { best_val = e; best_par = par; }
                return e;
            });
        opt_val = v; opt_par = p;
    } catch (...) {
        opt_val = best_val; opt_par = best_par;
    }

    auto counts = cudaq::sample(vqe_ansatz{}, n_qubits, reps, opt_par);
    return {counts.most_probable(), opt_val};
}

// Generic dispatch helpers
static GraphResult solve_qaoa(const Graph& g, const std::vector<double>& Q,
                               int n_qubits, int layers, int seed,
                               int (*decode)(const std::string&, const Graph&)) {
    check_qubit_limit(n_qubits, "QAOA");
    auto ising = qubo_to_ising(Q, n_qubits);
    auto [part, energy, opt_par] = run_qaoa(n_qubits, layers, ising, seed);
    return {part, decode(part, g), energy};
}

static GraphResult solve_vqe(const Graph& g, const std::vector<double>& Q,
                              int n_qubits, int reps, int seed,
                              int (*decode)(const std::string&, const Graph&)) {
    check_qubit_limit(n_qubits, "VQE");
    auto ising = qubo_to_ising(Q, n_qubits);
    auto [part, energy] = run_vqe(n_qubits, reps, ising, seed);
    return {part, decode(part, g), energy};
}

// MaxCut
GraphResult c2q_maxcut(const Graph& g, int layers, int seed) {
    return solve_qaoa(g, qubo_maxcut(g), g.num_nodes, layers, seed, decode_maxcut);
}
GraphResult c2q_maxcut_vqe(const Graph& g, int reps, int seed) {
    return solve_vqe(g, qubo_maxcut(g), g.num_nodes, reps, seed, decode_maxcut);
}

// MIS
GraphResult c2q_mis(const Graph& g, int layers, int seed) {
    return solve_qaoa(g, qubo_mis(g), g.num_nodes, layers, seed, decode_mis);
}
GraphResult c2q_mis_vqe(const Graph& g, int reps, int seed) {
    return solve_vqe(g, qubo_mis(g), g.num_nodes, reps, seed, decode_mis);
}

// Vertex Cover
GraphResult c2q_vc(const Graph& g, int layers, int seed) {
    return solve_qaoa(g, qubo_vc(g), g.num_nodes, layers, seed, decode_vc);
}
GraphResult c2q_vc_vqe(const Graph& g, int reps, int seed) {
    return solve_vqe(g, qubo_vc(g), g.num_nodes, reps, seed, decode_vc);
}

// Clique
GraphResult c2q_clique(const Graph& g, int k, int layers, int seed) {
    return solve_qaoa(g, qubo_clique(g, k), g.num_nodes, layers, seed, decode_clique);
}
GraphResult c2q_clique_vqe(const Graph& g, int k, int reps, int seed) {
    return solve_vqe(g, qubo_clique(g, k), g.num_nodes, reps, seed, decode_clique);
}

// KColor
// Decode wrapper that captures k
static int s_kcolor_k = 3;
static const Graph* s_kcolor_g = nullptr;
static int kcolor_decode_proxy(const std::string& bits, const Graph& g) {
    return decode_kcolor(bits, g, s_kcolor_k);
}

GraphResult c2q_kcolor(const Graph& g, int k, int layers, int seed) {
    int n_qubits = g.num_nodes * k;
    check_qubit_limit(n_qubits, "KColor QAOA");
    auto Q = qubo_kcolor(g, k);
    auto ising = qubo_to_ising(Q, n_qubits);
    auto [part, energy, opt_par] = run_qaoa(n_qubits, layers, ising, seed);
    return {part, decode_kcolor(part, g, k), energy};
}
GraphResult c2q_kcolor_vqe(const Graph& g, int k, int reps, int seed) {
    int n_qubits = g.num_nodes * k;
    check_qubit_limit(n_qubits, "KColor VQE");
    auto Q = qubo_kcolor(g, k);
    auto ising = qubo_to_ising(Q, n_qubits);
    auto [part, energy] = run_vqe(n_qubits, reps, ising, seed);
    return {part, decode_kcolor(part, g, k), energy};
}

// TSP
GraphResult c2q_tsp(const Graph& g, int layers, int seed) {
    int n_qubits = g.num_nodes * g.num_nodes;
    check_qubit_limit(n_qubits, "TSP QAOA");
    auto Q = qubo_tsp(g);
    auto ising = qubo_to_ising(Q, n_qubits);
    auto [part, energy, opt_par] = run_qaoa(n_qubits, layers, ising, seed);
    return {part, decode_tsp(part, g), energy};
}
GraphResult c2q_tsp_vqe(const Graph& g, int reps, int seed) {
    int n_qubits = g.num_nodes * g.num_nodes;
    check_qubit_limit(n_qubits, "TSP VQE");
    auto Q = qubo_tsp(g);
    auto ising = qubo_to_ising(Q, n_qubits);
    auto [part, energy] = run_vqe(n_qubits, reps, ising, seed);
    return {part, decode_tsp(part, g), energy};
}
