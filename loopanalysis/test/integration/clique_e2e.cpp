#include <algorithm>
#include <vector>
#include <iostream>
using namespace std;

// Same shape as test/clique.cpp / test/clique/basic.cpp: recursive
// max-clique backtracking with a running-best std::max update.

bool isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

int maxCliques(int start, vector<int>& clique, int size, int N,
              vector<vector<int>>& graph) {
  int best = 0;
  for (int v = start; v < N; v++) {
    clique[size] = v;
    if (isClique(size + 1, clique, graph)) {
      best = max(best, size + 1);
      best = max(best, maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

int findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return maxCliques(0, clique, 0, N, graph);
}

int main() {
  int N = 5;
  vector<vector<int>> graph(N, vector<int>(N, 0));

  // Triangle (0,1,2), node 3 adjacent to 0 only, node 4 isolated.
  // Max clique is the triangle -- size 3, not the trivial "everyone" answer.
  graph[0][1] = 1; graph[1][0] = 1;
  graph[1][2] = 1; graph[2][1] = 1;
  graph[0][2] = 1; graph[2][0] = 1;
  graph[0][3] = 1; graph[3][0] = 1;

  int expected = 3;

  cout << "Graph  : 5-vertex graph (triangle 0-1-2, pendant 3, isolated 4)\n";
  cout << "Solver : @clique_impl -> classical backtracking (no quantum kernel yet)\n\n";

  int result = findMaxClique(graph, N);

  cout << "Max clique size = " << result << "\n";
  cout << "Classical opt    = " << expected << "\n\n";

  if (result == expected) {
    cout << "PASS  exact optimum\n";
    return 0;
  }
  cout << "FAIL  result incorrect\n";
  return 1;
}
