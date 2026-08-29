#include <c2cudaq.h>
#include <c2cudaq/internal.h>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

using namespace c2cudaq;

// Bridge between the LLVM maxcut-cpp-pass and c2q_maxcut.
//
// The pass replaces a detected brute-force MaxCut loop nest with:
//
//   %r = call i32 @maxcut_impl(ptr %subsets, ptr %edges, ptr %best_S)
//
// where:
//   %subsets  - vector<vector<int>>* of precomputed subsets (ignored; QAOA
//               derives the partition space from qubit count, not enumeration)
//   %edges    - vector<pair<int,int>>* of graph edges (unweighted: w = 1.0)
//   %best_S   - vector<int>* to write the best partition into, or null
//
// Returns: cut value (number of crossing edges for the best partition found).
//
// c2q_maxcut has no invalid-answer mode at all -- every bitstring is a
// valid 2-partition (see analysis/maxcut/maxcut.md), unlike kcolor/
// clique/factor/tsp's kernels, which self-report failure via an
// unambiguous "no"/"-1". That means there's no decode-time signal here
// distinguishing "QAOA found the true maximum cut" from "QAOA settled
// for a valid-but-suboptimal one" -- the stress test found real cases of
// exactly that (e.g. random graphs collapsing to 0% optimal hit rate by
// N=14, still returning a perfectly *valid*, just not maximum, cut every
// time). So below a size where exhaustive enumeration stays cheap, this
// runs BOTH and keeps whichever is better -- a free correctness floor,
// not a guess.
namespace {

// 2^16 = 65536 partitions, trivially fast. Below this, run classical
// exact alongside the kernel and always keep the better cut; above it,
// kernel-only, as before -- the missing-invalid-signal risk documented
// above is deliberately left unaddressed past this point (a real
// tradeoff, not an oversight -- see analysis/maxcut/maxcut.md's next
// steps).
constexpr int kMaxCutExactCutoff = 16;

struct MaxCutExactResult {
  int value;
  unsigned mask;
};

MaxCutExactResult maxcut_impl_classical_exact(const Graph& g) {
  int best = 0;
  unsigned bestMask = 0;
  for (unsigned mask = 0; mask < (1u << g.num_nodes); ++mask) {
    int cut = 0;
    for (auto& [u, v, w] : g.edges)
      if (((mask >> u) & 1) != ((mask >> v) & 1))
        cut += static_cast<int>(w);
    if (cut > best) {
      best = cut;
      bestMask = mask;
    }
  }
  return {best, bestMask};
}

std::vector<int> maxcut_impl_mask_to_partition(unsigned mask, int n) {
  std::vector<int> nodes;
  for (int i = 0; i < n; ++i)
    if ((mask >> i) & 1)
      nodes.push_back(i);
  return nodes;
}

}  // namespace

extern "C" int maxcut_impl(const std::vector<std::vector<int>>* /*subsets*/,
                           const std::vector<std::pair<int, int>>* edges,
                           std::vector<int>* best_S) {
  // Build Graph from unweighted edge list.
  // num_nodes is derived as max endpoint index + 1.
  Graph g;
  g.num_nodes = 0;
  for (auto& [u, v] : *edges) {
    g.num_nodes = std::max(g.num_nodes, std::max(u, v) + 1);
    g.edges.emplace_back(u, v, 1.0);
  }
  if (g.num_nodes == 0 || g.edges.empty())
    return 0;

  if (g.num_nodes <= kMaxCutExactCutoff) {
    MaxCutExactResult exact = maxcut_impl_classical_exact(g);
    GraphResult r = c2q_maxcut(g);
    if (r.objective >= exact.value) {
      if (best_S)
        *best_S = decode_partition(r.partition);
      return r.objective;
    }
    if (best_S)
      *best_S = maxcut_impl_mask_to_partition(exact.mask, g.num_nodes);
    return exact.value;
  }

  GraphResult r = c2q_maxcut(g);

  if (best_S)
    *best_S = decode_partition(r.partition);

  return r.objective;
}

// Bridge between the LLVM kcolor-pass and c2q_kcolor.
//
// The pass replaces a detected top-level invocation of the m-coloring
// backtracking search -- solve(0, color, m, N, graph) in test/kcolor.cpp's
// shape -- with:
//
//   %r = call i1 @kcolor_impl(ptr %graph, i32 %m, i32 %N, ptr %color)
//
// where:
//   %graph - vector<vector<int>>* adjacency matrix (graph[u][v] == 1 iff
//            edge (u,v) exists)
//   %m     - number of colors to try (the "k" in k-coloring / m-coloring;
//            this is a caller-supplied FIXED value -- c2q_kcolor answers
//            "is m enough?", it does not search for the chromatic number)
//   %N     - number of nodes
//   %color - vector<int>* to write the discovered coloring into (0-indexed
//            colors), or null if the caller doesn't read it back
//
// Returns: true if the graph can be colored with at most m colors.
//
// c2q_kcolor (QAOA) is SOUND but not COMPLETE -- verified empirically
// (see loopanalysis/analysis/kcolor/): it never reports a false "valid"
// coloring, but frequently reports "invalid" even when a valid coloring
// exists (roughly 20-35% hit rate at the default layers=2, even on tiny
// 9-16 qubit graphs). So a kernel "true" can be trusted immediately; a
// kernel "false"/"-1" is ambiguous (truly infeasible, or QAOA just missed
// it) and must be verified classically before this function ever answers
// "not colorable" -- the kernel is a pure speed opportunist here, never a
// source of a wrong final answer.

namespace {

// Qubit count is N*m (one-hot encoding, one qubit per node-color pair).
// c2cudaq's own check_qubit_limit() throws above 28 qubits; this is a
// tighter, purely-performance cutoff well below that -- past this point
// the kernel is both slow (COBYLA over an increasingly large state space)
// and, per the stress test above, already unreliable enough that
// attempting it first rarely pays off before falling back to classical
// anyway.
constexpr int kKColorQubitSafeLimit = 24;

// Same algorithm as test/kcolor.cpp's isSafe()/solve()/graphColoring(),
// used as the guaranteed-correct fallback. Internally still tries colors
// 1..m with 0 reserved as an "unassigned" sentinel -- that's not just
// style, it's load-bearing: color[] is default-initialised to 0 and no
// real trial color is ever 0, so an as-yet-undecided node (k > node, still
// holding its initial value) can never spuriously collide with a trial
// color in isSafe()'s scan over all N nodes. classical_solve() below
// converts to the 0-indexed convention decode_kcolor_assignment() uses
// only at the very end, so color_out means the same thing regardless of
// which path (kernel or classical) produced it.
bool classical_is_safe(int node, const std::vector<int>& color,
                       const std::vector<std::vector<int>>& graph, int n,
                       int col) {
  for (int k = 0; k < n; ++k)
    if (k != node && graph[k][node] == 1 && color[k] == col)
      return false;
  return true;
}

bool classical_solve_rec(int node, std::vector<int>& color, int m, int N,
                         const std::vector<std::vector<int>>& graph) {
  if (node == N)
    return true;
  for (int i = 1; i <= m; ++i) {
    if (classical_is_safe(node, color, graph, N, i)) {
      color[node] = i;
      if (classical_solve_rec(node + 1, color, m, N, graph))
        return true;
      color[node] = 0;
    }
  }
  return false;
}

bool classical_solve(const std::vector<std::vector<int>>& graph, int m, int N,
                     std::vector<int>* color_out) {
  std::vector<int> color(N, 0);
  if (!classical_solve_rec(0, color, m, N, graph))
    return false;
  if (color_out) {
    color_out->assign(N, 0);
    for (int v = 0; v < N; ++v)
      (*color_out)[v] = color[v] - 1;  // 1..m -> 0..m-1
  }
  return true;
}

}  // namespace

extern "C" bool kcolor_impl(const std::vector<std::vector<int>>* graph,
                            int m, int N, std::vector<int>* color_out) {
  if (!graph)
    return false;

  if (N > 0 && m > 0 && N * m <= kKColorQubitSafeLimit) {
    Graph g;
    g.num_nodes = N;
    for (int u = 0; u < N; ++u)
      for (int v = u + 1; v < N; ++v)
        if ((*graph)[u][v] == 1)
          g.edges.emplace_back(u, v, 1.0);

    GraphResult r = c2q_kcolor(g, m);
    if (r.objective == 0) {
      if (color_out)
        *color_out = decode_kcolor_assignment(r.partition, N, m);
      return true;
    }
    // r.objective == -1 is ambiguous -- fall through to classical.
  }

  return classical_solve(*graph, m, N, color_out);
}

// Bridge between the LLVM tsp-pass and a classical solver, with a narrow
// kernel-first path.
//
// The pass replaces a detected brute-force TSP loop nest with:
//
//   %r = call i32 @tsp_impl(ptr %nodes, ptr %cost)
//
// where:
//   %nodes - vector<int>* of city indices to permute (only its size and
//            contents as a starting permutation matter -- the search
//            enumerates every permutation regardless of input order, since
//            it sorts first)
//   %cost  - vector<vector<int>>* cost matrix (cost[u][v] = edge weight)
//
// Returns: minimum tour cost (a full cycle, closing back to city 0).
//
// c2q_tsp (QAOA) was stress-tested (loopanalysis/analysis/tsp/tsp.md) and
// found to collapse fast: 4/10 correct at 3 total cities, 0/10 by 4 --
// far worse, far sooner, than any other kernel in this project. Also
// note (unlike kcolor/factor's kernel, whose validated answers are
// self-verifying, and only their NEGATIVE answers are ambiguous):
// decode_tsp validates a returned tour is a genuine Hamiltonian cycle,
// but that only proves the tour is VALID, not that it's the MINIMUM-cost
// one -- a valid-but-suboptimal tour would pass validation and silently
// return the wrong answer if trusted blindly.
//
// Both facts together leave exactly one size where kernel-first is both
// safe and worth attempting: 3 total cities. At N<=3, a complete graph
// has exactly ONE distinct Hamiltonian cycle (all 3 edges, either
// direction -- same total either way), so "valid" and "optimal"
// necessarily coincide; there is no second, cheaper tour a kernel could
// have missed. This is a genuinely degenerate property of N<=3, not a
// general one -- do NOT raise kTspQubitSafeMax without re-deriving this
// guarantee for the new N (it does not hold at N=4: multiple distinct
// Hamiltonian cycles exist there with different costs).
namespace {
constexpr int kTspQubitSafeMax = 3;
}  // namespace

extern "C" int tsp_impl(const std::vector<int>* nodes,
                        const std::vector<std::vector<int>>* cost) {
  if (!nodes || !cost || nodes->empty())
    return 0;

  int totalCities = static_cast<int>(nodes->size()) + 1;  // +1 for city 0
  if (totalCities <= kTspQubitSafeMax) {
    std::vector<int> cities;
    cities.push_back(0);
    for (int c : *nodes)
      cities.push_back(c);

    Graph g;
    g.num_nodes = totalCities;
    for (int i = 0; i < totalCities; ++i)
      for (int j = i + 1; j < totalCities; ++j)
        g.edges.emplace_back(
            i, j, static_cast<double>((*cost)[cities[i]][cities[j]]));

    GraphResult r = c2q_tsp(g);
    if (r.objective >= 0)
      return r.objective;
    // r.objective == -1 (no valid tour found) is ambiguous -- same
    // kernel-first-mandatory-classical-fallback-on-ambiguity convention
    // as kcolor_impl above -- fall through rather than trust a "no".
  }

  std::vector<int> perm = *nodes;
  std::sort(perm.begin(), perm.end());

  int minCost = std::numeric_limits<int>::max();
  do {
    int currCost = 0;
    int currNode = 0;
    for (std::size_t i = 0; i < perm.size(); ++i) {
      currCost += (*cost)[currNode][perm[i]];
      currNode = perm[i];
    }
    currCost += (*cost)[currNode][0];
    minCost = std::min(minCost, currCost);
  } while (std::next_permutation(perm.begin(), perm.end()));

  return minCost;
}

// Bridge between the LLVM clique-pass and a classical solver.
//
// The pass replaces a detected top-level invocation of the max-clique
// backtracking search -- maxCliques(0, clique, 0, N, graph) in
// test/clique.cpp's shape -- with:
//
//   %r = call i32 @clique_impl(ptr %graph, i32 %N, ptr %clique)
//
// where:
//   %graph  - vector<vector<int>>* adjacency matrix
//   %N      - number of vertices
//   %clique - vector<int>* to write the largest clique's members into
//             (resized to the returned size), or null if unused
//
// Returns: size of the largest clique found. Classical recursive
// backtracking -- same algorithm as test/clique.cpp's maxCliques()/
// isClique(). No quantum kernel yet, same reasoning as tsp_impl above:
// c2q_clique's own stress test (analysis/clique/) already found reliability
// and wall-clock limits well inside its 28-qubit hard cap, and wiring a
// kernel-first path in here is separate, larger work (see
// loopanalysis/analysis/clique/clique.md's next-steps) -- this bridge
// exists so clique-pass has something to link against today.
namespace {

bool clique_impl_is_clique(int size, const std::vector<int>& clique,
                           const std::vector<std::vector<int>>& graph) {
  for (int i = 0; i < size; ++i)
    for (int j = i + 1; j < size; ++j)
      if (graph[clique[i]][clique[j]] == 0)
        return false;
  return true;
}

// bestClique is threaded by reference through the whole recursion and
// updated in place whenever any level (this one or a deeper one) finds a
// new best, so it always holds the best clique found so far by the time
// the top-level call returns. Whether to overwrite it is decided against
// bestClique.size() itself -- the one value every stack frame shares and
// that always reflects the true global best -- not against `best`, which
// is local to each frame and starts at 0 on every call. Comparing against
// `best` instead (an earlier version of this function did) lets a shallow
// frame that only found a small extension overwrite bestClique with a
// worse answer after a deeper frame already found a better one; the
// *size* returned would still come out correct, only the recorded
// membership would be wrong -- worth calling out since it's an easy bug to
// reintroduce and the size-only return would never catch it.
int clique_impl_max_cliques(int start, std::vector<int>& clique, int size,
                            int N, const std::vector<std::vector<int>>& graph,
                            std::vector<int>& bestClique) {
  int best = 0;
  for (int v = start; v < N; ++v) {
    clique[size] = v;
    if (clique_impl_is_clique(size + 1, clique, graph)) {
      if (size + 1 > best)
        best = size + 1;
      if (size + 1 > static_cast<int>(bestClique.size()))
        bestClique.assign(clique.begin(), clique.begin() + size + 1);
      int extended = clique_impl_max_cliques(v + 1, clique, size + 1, N,
                                             graph, bestClique);
      if (extended > best)
        best = extended;
    }
  }
  return best;
}

}  // namespace

extern "C" int clique_impl(const std::vector<std::vector<int>>* graph, int N,
                           std::vector<int>* clique_out) {
  if (!graph || N <= 0)
    return 0;

  std::vector<int> clique(N, 0);
  std::vector<int> bestClique;
  int best = clique_impl_max_cliques(0, clique, 0, N, *graph, bestClique);

  if (clique_out)
    *clique_out = bestClique;
  return best;
}

// Bridge between the LLVM factor-pass and a solver, kernel-first within
// c2q_factor's verified-safe range.
//
// The pass replaces a detected brute-force factor-pair search loop nest --
// bruteForceFactor(n, outA, outB) in test/factor.cpp's shape -- with:
//
//   %r = call i1 @factor_impl(i32 %n, ptr %outA, ptr %outB)
//
// where:
//   %n    - the number to factor
//   %outA - int* to write the first factor into
//   %outB - int* to write the second factor into (outA * outB == n on
//           success; (1, n) on failure, i.e. n is prime or n < 4)
//
// Returns: true if a non-trivial factor pair was found.
//
// c2q_factor (Grover) was stress-tested (loopanalysis/analysis/factor/
// factor.md) and found correct in EVERY trial across its entire n<=127
// reachable range (26/26, composites and primes both) -- unlike the
// other four kernels in this project, no false or ambiguous-but-wrong
// answer was ever observed. It's also 5-10 orders of magnitude slower
// than the classical loop it replaces at every size in that range on a
// state-vector simulator, which is why this bridge stayed classical-only
// for a while (see factor.md's verdict) -- but per an explicit project
// decision, wall-clock cost is not the deciding factor here: the point
// is to actually exercise the quantum path within its verified-correct
// range, with classical staying only as the fallback and the
// out-of-range case, not the default. Revisit the cutoff, not the
// kernel-first choice itself, once non-simulated hardware changes the
// wall-clock calculus.
namespace {
// Grover's oracle needs total_q = 4*(bitlen(n)-1)+1 qubits; c2q_factor
// itself throws above 28 (see c2cudaq/include/c2cudaq/internal.h's
// check_qubit_limit). n=127 -> bitlen=7 -> total_q=25 (safe); n=128 ->
// bitlen=8 -> total_q=29 (throws) -- 127 is exactly the largest n that
// stays under the limit, matching the range the stress test covered.
constexpr int kFactorQubitSafeMax = 127;
}  // namespace

extern "C" bool factor_impl(int n, int* outA, int* outB) {
  if (!outA || !outB)
    return false;

  if (n >= 4 && n <= kFactorQubitSafeMax) {
    auto [a, b] = c2q_factor(n);
    // Only a POSITIVE claim (a genuine factor pair) is trusted without
    // falling through -- self-verified for free via a*b==n. An a==1 "no
    // factor found" answer is ambiguous (n could be genuinely prime, or
    // the kernel could have missed an existing factor) and, matching
    // this project's kernel-first-mandatory-classical-fallback-on-
    // ambiguity convention (see kcolor_impl above), is never trusted
    // alone -- falls through to the exhaustive search either way. In
    // practice the stress test found this branch never actually needed
    // (0 missed factors, 0 false positives across 26 trials), but the
    // guard costs nothing and matches this project's established
    // pattern for every other kernel.
    if (a > 1 && b > 1 && a * b == n) {
      *outA = static_cast<int>(a);
      *outB = static_cast<int>(b);
      return true;
    }
  }

  for (int a = 2; a < n; ++a) {
    for (int b = 2; b < n; ++b) {
      if (a * b == n) {
        *outA = a;
        *outB = b;
        return true;
      }
    }
  }
  *outA = 1;
  *outB = n;
  return false;
}
