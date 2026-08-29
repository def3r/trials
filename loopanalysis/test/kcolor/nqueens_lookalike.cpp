// nqueens_lookalike.cpp — N-Queens-with-obstacles, NOT m-coloring, reshaped
// to fit the 5-argument pattern matchSolve() looks for
//
// This is a genuinely different problem: place a queen per row so no two
// attack each other, avoiding pre-blocked squares, trying at most
// `maxCols` columns per row. Natural N-Queens has only 3 arguments (row,
// board, N) -- board size (N) and column count are the same quantity, so
// matchSolve()'s MArg search (which explicitly excludes NArg) would reject
// it outright. Reshaping it to take a REDUNDANT, separately-named column
// limit (`maxCols`, distinct from N even though it would equal N in any
// real use) and an obstacle matrix (to give it something GraphArg-shaped)
// produces a function occupying the same 5-argument-role pattern kcolor
// does: node, index-addressed container, two distinct bound arguments, one
// more container threaded into the guard call.
//
// This DOES currently match and get replaced -- confirmed by running the
// pass, not assumed. matchSolve() has no way to distinguish "assigns a
// column per row, backtracking on a diagonal/column/obstacle conflict"
// from "assigns a color per node, backtracking on an adjacency conflict":
// both compile to an identical shape (self-call, node+1, a container
// indexed by node with an assign/backtrack store pair, a guard call taking
// node and a second container, a candidate loop bounded by a third
// argument distinct from N). If replaced, the pass hands `obstacles` to
// c2q_kcolor as if it were a graph adjacency matrix and answers a
// completely different question than the one this code asks -- a silent
// wrong-answer bug, not just a missed optimization.
//
// This is left as a documented, known limitation (not fixed here): telling
// "graph adjacency check" apart from "arbitrary per-cell constraint check"
// from IR shape alone, without deeper semantic reasoning about what the
// guard call's second container actually represents, looks like a genuine
// limit of purely-structural matching rather than a narrow oversight with
// an easy fix.
//
// Extract target: kc_nqueens_ (isSafe, solve, place)

#include <vector>
using namespace std;

bool kc_nqueens_isSafe(int row, vector<int>& board,
                       vector<vector<int>>& obstacles, int N, int col) {
  if (obstacles[row][col] == 1) {
    return false;
  }
  for (int r = 0; r < row; r++) {
    int c = board[r];
    int rowDiff = r - row;
    int colDiff = c - col;
    if (rowDiff < 0)
      rowDiff = -rowDiff;
    if (colDiff < 0)
      colDiff = -colDiff;
    if (c == col || rowDiff == colDiff) {
      return false;
    }
  }
  return true;
}

bool kc_nqueens_solve(int row, vector<int>& board, int maxCols, int N,
                      vector<vector<int>>& obstacles) {
  if (row == N) {
    return true;
  }
  for (int col = 0; col < maxCols; col++) {
    if (kc_nqueens_isSafe(row, board, obstacles, N, col)) {
      board[row] = col;
      if (kc_nqueens_solve(row + 1, board, maxCols, N, obstacles))
        return true;
      board[row] = -1;
    }
  }
  return false;
}

bool kc_nqueens_place(vector<vector<int>>& obstacles, int maxCols, int N) {
  vector<int> board(N, -1);
  if (kc_nqueens_solve(0, board, maxCols, N, obstacles))
    return true;
  return false;
}

// --- lit check directives (read by update.py) ---
// Expected: NOT detected -- but XFAIL: currently IS detected (see above).
// XFAIL: *
// CHECK-NOT: kcolor_impl
