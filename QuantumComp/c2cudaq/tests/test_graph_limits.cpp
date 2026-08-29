// Scaling/limits probe for the MaxCut kernel (c2q_maxcut / c2q_maxcut_vqe).
//
// Not part of the default `ctest` run (see tests/CMakeLists.txt — this test
// carries the "slow" label). It sweeps node count and edge density, compares
// the QAOA/VQE result against a classical brute-force optimum, and reports
// where approximation quality starts to degrade. No hard failures on quality
// loss (that's expected heuristic behavior) — only on crashes or a partition
// the decoder marks invalid (objective == -1), which indicates a real bug.
//
// Run explicitly:
//   ctest --test-dir build -L slow --output-on-failure
// or directly:
//   ./build/tests/test_graph_limits [max_nodes]
#include <c2cudaq.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static int failures = 0;

static Graph make_random_graph(int n, double edge_prob, unsigned seed) {
  Graph g{n, {}};
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (coin(rng) < edge_prob)
        g.edges.emplace_back(i, j, 1.0);
  return g;
}

// Exact MaxCut via brute force over all 2^n partitions. Only tractable for
// small n (used here up to ~20 nodes: 2^20 masks, cheap).
static int brute_force_maxcut(const Graph& g) {
  int best = 0;
  int n = g.num_nodes;
  for (uint32_t mask = 0; mask < (1u << n); ++mask) {
    int cut = 0;
    for (auto& [u, v, w] : g.edges) {
      bool su = (mask >> u) & 1;
      bool sv = (mask >> v) & 1;
      if (su != sv)
        cut += static_cast<int>(w);
    }
    if (cut > best)
      best = cut;
  }
  return best;
}

struct Sample {
  int nodes;
  int edges;
  int optimal;
  int qaoa_cut;
  int vqe_cut;
  double qaoa_ratio;
  double vqe_ratio;
  double qaoa_seconds;
  double vqe_seconds;
};

static Sample run_sample(const Graph& g) {
  Sample s{};
  s.nodes = g.num_nodes;
  s.edges = static_cast<int>(g.edges.size());
  s.optimal = brute_force_maxcut(g);

  auto t0 = std::chrono::steady_clock::now();
  auto mc = c2q_maxcut(g);
  auto t1 = std::chrono::steady_clock::now();
  auto mc_vqe = c2q_maxcut_vqe(g);
  auto t2 = std::chrono::steady_clock::now();

  s.qaoa_cut = mc.objective;
  s.vqe_cut = mc_vqe.objective;
  s.qaoa_seconds = std::chrono::duration<double>(t1 - t0).count();
  s.vqe_seconds = std::chrono::duration<double>(t2 - t1).count();

  if (mc.objective < 0) {
    std::cout << "[FAIL] QAOA MaxCut N=" << s.nodes
              << ": decoder reported invalid partition (\"" << mc.partition
              << "\")\n";
    ++failures;
    s.qaoa_ratio = 0.0;
  } else {
    s.qaoa_ratio = s.optimal > 0
                       ? static_cast<double>(s.qaoa_cut) / s.optimal
                       : 1.0;
  }

  if (mc_vqe.objective < 0) {
    std::cout << "[FAIL] VQE MaxCut N=" << s.nodes
              << ": decoder reported invalid partition (\""
              << mc_vqe.partition << "\")\n";
    ++failures;
    s.vqe_ratio = 0.0;
  } else {
    s.vqe_ratio = s.optimal > 0
                      ? static_cast<double>(s.vqe_cut) / s.optimal
                      : 1.0;
  }

  return s;
}

static void print_row(const Sample& s) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "[INFO] N=" << std::setw(2) << s.nodes
            << " E=" << std::setw(3) << s.edges
            << " optimal=" << std::setw(3) << s.optimal
            << " | QAOA cut=" << std::setw(3) << s.qaoa_cut
            << " ratio=" << s.qaoa_ratio << " (" << s.qaoa_seconds << "s)"
            << " | VQE cut=" << std::setw(3) << s.vqe_cut
            << " ratio=" << s.vqe_ratio << " (" << s.vqe_seconds << "s)"
            << "\n";
}

// Reports (does not fail on) the first sample where ratio drops below
// `threshold`, scanning in the order samples were collected.
static void report_degradation_point(const char* label,
                                      const std::vector<Sample>& samples,
                                      double Sample::*ratio_field,
                                      double threshold) {
  for (auto& s : samples) {
    if (s.*ratio_field < threshold) {
      std::cout << "[INFO] " << label << " approx ratio first drops below "
                << threshold << " at N=" << s.nodes << " E=" << s.edges
                << " (ratio=" << (s.*ratio_field) << ")\n";
      return;
    }
  }
  std::cout << "[INFO] " << label << " approx ratio stayed >= " << threshold
            << " across the whole sweep\n";
}

int main(int argc, char** argv) {
  // Cap kept conservative for a small (e.g. 4GB) GPU. Raise via argv[1] on
  // more capable hardware to push closer to the README's N<=28 claim.
  int max_nodes = 20;
  if (argc > 1)
    max_nodes = std::atoi(argv[1]);

  std::cout << "== MaxCut node-count sweep (fixed edge_prob=0.4) ==\n";
  std::vector<Sample> node_sweep;
  for (int n = 4; n <= max_nodes; n += 2) {
    Graph g = make_random_graph(n, 0.4, /*seed=*/1000 + n);
    Sample s = run_sample(g);
    print_row(s);
    node_sweep.push_back(s);
  }
  report_degradation_point("Node sweep / QAOA", node_sweep, &Sample::qaoa_ratio,
                            0.9);
  report_degradation_point("Node sweep / VQE", node_sweep, &Sample::vqe_ratio,
                            0.9);

  // Fixed node count, sweep from sparse (tree) to dense (complete graph) to
  // see how edge count / circuit depth affects approximation quality.
  int density_n = std::min(14, max_nodes);
  std::cout << "\n== MaxCut edge-density sweep (fixed N=" << density_n
            << ") ==\n";
  std::vector<Sample> edge_sweep;
  std::vector<double> probs = {0.15, 0.3, 0.5, 0.7, 0.85, 1.0};
  for (double p : probs) {
    Graph g = make_random_graph(density_n, p, /*seed=*/2000);
    Sample s = run_sample(g);
    print_row(s);
    edge_sweep.push_back(s);
  }
  report_degradation_point("Edge sweep / QAOA", edge_sweep, &Sample::qaoa_ratio,
                            0.9);
  report_degradation_point("Edge sweep / VQE", edge_sweep, &Sample::vqe_ratio,
                            0.9);

  std::cout << "\n"
            << (failures == 0
                    ? "All graph limit probes completed with valid partitions."
                    : std::to_string(failures) + " invalid partition(s) detected.")
            << "\n";
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
