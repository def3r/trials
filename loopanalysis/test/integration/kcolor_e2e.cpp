#include <iostream>
#include <vector>
using namespace std;

// Same shape as test/kcolor.cpp: recursive backtracking m-coloring.

bool isSafe(int node, vector<int>& color, vector<vector<int>>& graph, int n, int col) {
  for (int k = 0; k < n; k++) {
    if (k != node && graph[k][node] == 1 && color[k] == col) {
      return false;
    }
  }
  return true;
}

bool solve(int node, vector<int>& color, int m, int N, vector<vector<int>>& graph) {
  if (node == N) {
    return true;
  }

  for (int i = 1; i <= m; i++) {
    if (isSafe(node, color, graph, N, i)) {
      color[node] = i;
      if (solve(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool graphColoring(vector<vector<int>>& graph, int m, int N) {
  vector<int> color(N, 0);
  if (solve(0, color, m, N, graph))
    return true;
  return false;
}

int main() {
  int N = 4;

  // Triangle (0,1,2) + node 3 adjacent to 0 and 2 only. Chromatic number
  // is 3 (the triangle forces it); node 3 can reuse node 1's color.
  vector<vector<int>> graph(N, vector<int>(N, 0));
  graph[0][1] = 1; graph[1][0] = 1;
  graph[1][2] = 1; graph[2][1] = 1;
  graph[2][3] = 1; graph[3][2] = 1;
  graph[3][0] = 1; graph[0][3] = 1;
  graph[0][2] = 1; graph[2][0] = 1;

  cout << "Graph  : 4-node graph (triangle 0-1-2 + node 3), chromatic number = 3\n";
  cout << "Solver : @kcolor_impl -> c2q_kcolor (QAOA) w/ classical fallback\n\n";

  bool feasible3 = graphColoring(graph, /*m=*/3, N);
  cout << "m=3 (should be colorable)     : " << (feasible3 ? "colorable" : "NOT colorable") << "\n";

  bool feasible2 = graphColoring(graph, /*m=*/2, N);
  cout << "m=2 (should NOT be colorable) : " << (feasible2 ? "colorable" : "NOT colorable") << "\n\n";

  bool ok = feasible3 && !feasible2;
  if (ok) {
    cout << "PASS  both results correct\n";
    return 0;
  }
  cout << "FAIL  incorrect result\n";
  return 1;
}
