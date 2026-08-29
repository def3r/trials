#include <c2cudaq.h>
#include <iostream>

static void print_result(const char* label, const GraphResult& r) {
  std::cout << label << "\n"
            << "  partition : " << r.partition << "\n"
            << "  objective : " << r.objective << "\n"
            << "  energy    : " << r.energy << "\n";
}

int main() {
  std::cout << "=== Quantum Graph Optimization Examples ===\n\n";

  // Five-node graph used in maxcut.py
  //   v1 - v2
  //   |  / |
  //   v0 - v3 - v4
  Graph g5{5,
           {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 0, 1}, {2, 4, 1}, {3, 4, 1}}};

  // MaxCut
  std::cout << "-- MaxCut QAOA (layers=2, default) --\n";
  print_result("MaxCut", c2q_maxcut(g5));

  std::cout << "\n-- MaxCut QAOA (layers=3, deeper) --\n";
  print_result("MaxCut-3", c2q_maxcut(g5, /*layers=*/3));

  std::cout << "\n-- MaxCut VQE (reps=2) --\n";
  print_result("MaxCut-VQE", c2q_maxcut_vqe(g5));

  // MIS
  // Path graph P4: 0-1-2-3, optimal MIS = size 2
  Graph path4{4, {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}}};

  std::cout << "\n-- MIS QAOA (path P4) --\n";
  print_result("MIS", c2q_mis(path4));

  std::cout << "\n-- MIS VQE (path P4) --\n";
  print_result("MIS-VQE", c2q_mis_vqe(path4));

  // Vertex Cover
  // Star K_{1,3}: center=0, leaves=1,2,3 - min VC = {0}
  Graph star{4, {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}}};

  std::cout << "\n-- Vertex Cover QAOA (star K_1,3) --\n";
  print_result("VC", c2q_vc(star));

  // Clique
  // K4 complete graph - largest clique = 4
  Graph k4{4,
           {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 2, 1}, {1, 3, 1}, {2, 3, 1}}};

  std::cout << "\n-- Clique QAOA (K4, target k=2) --\n";
  print_result("Clique", c2q_clique(k4, /*k=*/2));

  // KColor
  // Triangle K3 - needs 3 colors
  Graph k3{3, {{0, 1, 1}, {1, 2, 1}, {0, 2, 1}}};

  std::cout << "\n-- KColor QAOA (triangle K3, k=3) --\n";
  auto kc = c2q_kcolor(k3, /*k=*/3);
  std::cout << "KColor\n  partition : " << kc.partition
            << "\n  valid     : " << (kc.objective == 0 ? "yes" : "no")
            << "\n  energy    : " << kc.energy << "\n";

  // TSP
  // 3-city complete weighted graph
  Graph tsp3{3, {{0, 1, 2}, {1, 2, 3}, {0, 2, 4}}};
  // {0, 10, 15, 20}, {10, 0, 35, 25}, {15, 35, 0, 30}, {20, 25, 30, 0}
  Graph tsp4{
      4,
      {{0, 1, 10}, {0, 2, 15}, {0, 3, 20}, {1, 2, 35}, {1, 3, 25}, {2, 3, 30}}};

  std::cout << "\n-- TSP QAOA (3 cities) --\n";
  auto tsp = c2q_tsp(tsp3);
  std::cout << "TSP\n  partition  : " << tsp.partition
            << "\n  tour_dist  : " << tsp.objective << " ("
            << (tsp.objective < 0 ? "invalid tour" : "valid tour") << ")"
            << "\n  energy     : " << tsp.energy << "\n";

  std::cout << "\n-- TSP QAOA (4 cities) --\n";
  tsp = c2q_tsp(tsp4);
  std::cout << "TSP\n  partition  : " << tsp.partition
            << "\n  tour_dist  : " << tsp.objective << " ("
            << (tsp.objective < 0 ? "invalid tour" : "valid tour") << ")"
            << "\n  energy     : " << tsp.energy << "\n";

  return 0;
}
