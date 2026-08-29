#include <iostream>
#include "libmaxcut.h"

int main() {
  // Same graph as maxcut.py
  //   v1--v2
  //   |  /| \
  //   | / |  \
  //   v0--v3--v4
  int num_nodes = 4;
  std::vector<std::pair<int, int>> edges = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

  auto result = solve_maxcut(num_nodes, edges, 2);

  std::cout << "Partition : " << result.partition << "\n";
  std::cout << "Cut value : " << result.cut_value << "\n";
  std::cout << "Optimal E : " << result.optimal_energy << "\n";
  std::cout << "Max cut >= " << -result.optimal_energy << "\n";
  return 0;
}
