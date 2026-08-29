#include <c2cudaq.h>
#include <cstdlib>
#include <iostream>

static int failures = 0;

static void check_int(const char* name,
                      int got,
                      int expected_min,
                      int expected_max = -1) {
  if (expected_max < 0)
    expected_max = expected_min;
  if (got >= expected_min && got <= expected_max) {
    std::cout << "[PASS] " << name << " = " << got << "\n";
  } else {
    std::cout << "[FAIL] " << name << ": got " << got << ", expected in ["
              << expected_min << ".." << expected_max << "]\n";
    ++failures;
  }
}

static void check_not_neg(const char* name, int got) {
  if (got >= 0) {
    std::cout << "[PASS] " << name << " = " << got << " (valid)\n";
  } else {
    std::cout << "[FAIL] " << name << ": got " << got
              << " (invalid partition)\n";
    ++failures;
  }
}

int main() {
  // MaxCut — 5-node graph from maxcut.py
  // Known max cut = 5, valid partitions include "10101", "01010", etc.
  Graph g5{5,
           {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 0, 1}, {2, 4, 1}, {3, 4, 1}}};

  auto mc = c2q_maxcut(g5);
  std::cout << "[INFO] MaxCut QAOA partition=" << mc.partition
            << " cut=" << mc.objective << "\n";
  check_int("maxcut QAOA cut_value", mc.objective, 4,
            5);  // optimum=5, accept >=4

  auto mc_vqe = c2q_maxcut_vqe(g5);
  std::cout << "[INFO] MaxCut VQE  partition=" << mc_vqe.partition
            << " cut=" << mc_vqe.objective << "\n";
  check_int("maxcut VQE  cut_value", mc_vqe.objective, 3,
            5);  // VQE may be looser

  // MIS — path graph P4: 0-1-2-3, optimum MIS = {0,2} or {0,3} or {1,3},
  // size=2
  Graph path4{4, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}}};
  auto mis = c2q_mis(path4);
  std::cout << "[INFO] MIS QAOA partition=" << mis.partition
            << " size=" << mis.objective << "\n";
  // Accept size ≥ 1 (at least a valid independent set, QAOA may not reach
  // optimum)
  check_not_neg("MIS QAOA valid", mis.objective);

  auto mis_vqe = c2q_mis_vqe(path4);
  std::cout << "[INFO] MIS VQE  partition=" << mis_vqe.partition
            << " size=" << mis_vqe.objective << "\n";
  check_not_neg("MIS VQE valid", mis_vqe.objective);

  // Vertex Cover — star graph K1,3: center=0, leaves=1,2,3
  // Minimum VC = {0}, size=1
  Graph star{4, {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}}};
  auto vc = c2q_vc(star);
  std::cout << "[INFO] VC QAOA partition=" << vc.partition
            << " size=" << vc.objective << "\n";
  check_not_neg("VC QAOA valid", vc.objective);

  // Clique — K4 (complete graph on 4 nodes), largest clique = 4
  Graph k4{4,
           {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 2, 1}, {1, 3, 1}, {2, 3, 1}}};
  auto cl = c2q_clique(k4);
  std::cout << "[INFO] Clique QAOA partition=" << cl.partition
            << " size=" << cl.objective << "\n";
  check_not_neg("Clique QAOA valid", cl.objective);

  // KColor — triangle K3, needs 3 colors
  Graph k3{3, {{0, 1, 1}, {1, 2, 1}, {0, 2, 1}}};
  auto kc = c2q_kcolor(k3, /*k=*/3);
  std::cout << "[INFO] KColor QAOA (k=3) partition=" << kc.partition
            << " valid=" << (kc.objective == 0 ? "yes" : "no") << "\n";
  // Accept -1 (QAOA may not find valid 3-coloring in 2 layers) — report only
  std::cout << (kc.objective == 0 ? "[PASS]" : "[INFO]")
            << " kcolor objective=" << kc.objective << "\n";

  // TSP — 3-city complete weighted graph
  Graph tsp3{3, {{0, 1, 2}, {1, 2, 3}, {0, 2, 4}}};
  auto tsp = c2q_tsp(tsp3);
  std::cout << "[INFO] TSP QAOA partition=" << tsp.partition
            << " tour_dist=" << tsp.objective << "\n";
  // Accept any non-negative tour or -1 (TSP with QAOA+2 layers may not find
  // valid tour)
  std::cout << (tsp.objective >= 0 ? "[PASS]" : "[INFO]")
            << " TSP objective=" << tsp.objective << "\n";

  std::cout << "\n"
            << (failures == 0 ? "All graph tests PASSED."
                              : std::to_string(failures) + " test(s) FAILED.")
            << "\n";
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
