// C++ implementation of maximum clique search via recursive extension.
//
// Same algorithm as the original (find the largest clique by repeatedly
// trying to extend the current one with a later vertex), rewritten to
// match the input-passing convention used by tsp.cpp/kcolor.cpp: a
// vector<vector<int>> adjacency matrix and vector<int> working state
// passed by reference, no global mutable arrays. That's the main thing
// making an LLVM IR matcher tractable at all -- global arrays show up as
// @graph/@store references with no argument-based container to trace,
// where vector<vector<int>>& shows up as a clean operator[]-call chain the
// same way maxcut_pass.cpp/tsp_pass.cpp/kcolor_pass.cpp already know how to
// recognise.
//
// Note for whoever writes clique_pass.cpp: the recursion here advances via
// the LOOP variable (maxCliques(v + 1, ...), where v ranges over the
// for-loop), not via the function's own `start` parameter + 1 the way
// kcolor's solve(node + 1, ...) does. That's inherent to "extend the
// clique starting after whichever vertex we just added" -- kcolor_pass's
// self-recursion matcher (which specifically requires the self-call's
// argument to be FormalArg + 1) won't recognise this shape as-is; a clique
// matcher needs to look for FormalArg's recursion argument being the
// CANDIDATE LOOP'S OWN INDUCTION VARIABLE + 1 instead.

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// Function to check if the given set of vertices in clique[0..size-1] is a
// clique (every pair is connected).
bool isClique(int size, vector<int>& clique, vector<vector<int>>& graph) {
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      // If any edge is missing
      if (graph[clique[i]][clique[j]] == 0) {
        return false;
      }
    }
  }
  return true;
}

// Function to find the size of the largest clique reachable by extending
// clique[0..size-1] with vertices from `start` onward.
int maxCliques(int start,
               vector<int>& clique,
               int size,
               int N,
               vector<vector<int>>& graph) {
  int best = 0;

  // Check if any vertex from start onward can be added
  for (int v = start; v < N; v++) {
    // Add the vertex to clique
    clique[size] = v;

    // If clique[0..size] is not itself a clique, extending it further
    // with v can't be one either
    if (isClique(size + 1, clique, graph)) {
      // Update best with the clique we just confirmed
      best = max(best, size + 1);

      // Check if it can be extended further
      best = max(best, maxCliques(v + 1, clique, size + 1, N, graph));
    }
  }
  return best;
}

// Entry point: find the size of the maximum clique in `graph`.
int findMaxClique(vector<vector<int>>& graph, int N) {
  vector<int> clique(N, 0);
  return maxCliques(0, clique, 0, N, graph);
}

int main() {
  int N = 4;
  vector<vector<int>> graph(N, vector<int>(N, 0));

  // Complete graph K4 -- every pair of vertices is connected, so the
  // maximum clique is the whole graph (size 4).
  graph[0][1] = 1;
  graph[1][0] = 1;
  graph[1][2] = 1;
  graph[2][1] = 1;
  graph[2][0] = 1;
  graph[0][2] = 1;
  graph[3][2] = 1;
  graph[2][3] = 1;
  graph[3][0] = 1;
  graph[0][3] = 1;
  graph[3][1] = 1;
  graph[1][3] = 1;

  cout << findMaxClique(graph, N) << endl;

  return 0;
}
