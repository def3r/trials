# c2cudaq

C++ library for quantum-classical optimization and arithmetic using CUDA-Q (nvq++).
Covers 6 graph optimization problems (QAOA + VQE) and quantum arithmetic (ADD, SUB, MUL, FACTOR).

## Build

```bash
cmake -DCMAKE_CXX_COMPILER=/path/to/c2cudaq/nvqpp_wrap.sh \
      -DCMAKE_CXX_COMPILER_FORCED=TRUE \
      -B build -S .
cmake --build build -j$(nproc)
```

The `nvqpp_wrap.sh` wrapper strips CMake's `-MD/-MF` dependency-file flags that nvq++ does not support.

### Run tests

```bash
LD_LIBRARY_PATH=/opt/cuda-13.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
    ctest --test-dir build --output-on-failure -LE slow
```

`test_graph_limits` (MaxCut node-count / edge-density scaling probe, see below)
is labeled `slow` and excluded above. Run it explicitly:

```bash
LD_LIBRARY_PATH=/opt/cuda-13.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH \
    ctest --test-dir build --output-on-failure -L slow
# or, to raise the node cap above the default 20 (GPU-memory permitting):
./build/tests/test_graph_limits 26
```

## API

```cpp
#include <c2cudaq.h>
```

### Arithmetic

| Function | Description | Qubit limit |
|---|---|---|
| `c2q_add(a, b)` | Quantum ripple-carry addition | ~2·ceil(log₂(max(a,b)+1))+1 |
| `c2q_sub(a, b)` | Two's-complement subtraction | same as add |
| `c2q_mul(a, b)` | QFT multiply (auto bit widths) | 4·ceil(log₂(max(a,b)+1)) |
| `c2q_mul(a, b, sa, sb)` | QFT multiply (explicit widths) | 2·(sa+sb) |
| `c2q_factor(n)` | Quantum-verified factoring | 4·ceil(log₂(n)/2) |

**Limits:** The simulator holds a 2^N statevector in memory.  28 qubits ≈ 4 GB RAM.
Safe practical ranges (statevector sim):
- ADD/SUB: up to ~13-bit operands (28 qubits)
- MUL (auto): operands up to ~2000 (uses `ceil(log₂)` bits each)
- MUL (explicit sa, sb): constrained by 2*(sa+sb) ≤ 28
- FACTOR: n < 2^14 (14-bit result); factor circuit needs 4·sa+4·sb qubits

### Graph Problems

All graph functions accept a `Graph` struct and return `GraphResult`:

```cpp
struct Graph {
    int num_nodes;
    std::vector<std::tuple<int,int,double>> edges;  // (u, v, weight)
};

struct GraphResult {
    std::string partition;  // bitstring: bit i = 1 means node i is selected
    int         objective;  // problem-specific objective value (-1 if invalid)
    double      energy;     // final QAOA/VQE energy
};
```

| Function | Problem | Qubits | Overloads |
|---|---|---|---|
| `c2q_maxcut(g, layers=2, seed=13)` | Max-Cut | N | `c2q_maxcut_vqe(g, reps=2, seed=13)` |
| `c2q_mis(g, layers=2, seed=13)` | Max Independent Set | N | `c2q_mis_vqe(...)` |
| `c2q_vc(g, layers=2, seed=13)` | Min Vertex Cover | N | `c2q_vc_vqe(...)` |
| `c2q_clique(g, k=-1, layers=2, seed=13)` | Max Clique | N | `c2q_clique_vqe(...)` |
| `c2q_kcolor(g, k=3, layers=2, seed=13)` | k-Coloring | N·k | `c2q_kcolor_vqe(...)` |
| `c2q_tsp(g, layers=2, seed=13)` | Traveling Salesman | N² | `c2q_tsp_vqe(...)` |

**Qubit limits per problem:**
- MaxCut / MIS / VC / Clique: N qubits → practical limit **N ≤ 28 nodes**
- KColor: N·k qubits → practical limit **N·k ≤ 28** (e.g. 9 nodes, k=3)
- TSP: N² qubits → practical limit **N ≤ 5 cities**

The N ≤ 28 figure above is a memory ceiling (statevector fits in GPU RAM), not
a quality guarantee. `tests/test_graph_limits.cpp` sweeps MaxCut node count and
edge density against a classical brute-force optimum and reports the
approximation ratio at each size, so you can see where QAOA/VQE result quality
actually starts to drop rather than just where the simulator runs out of
memory. It's opt-in (see "Run tests" above) since it's much slower than the
rest of the suite.

**Simulator only:** All functions use the CUDA-Q statevector simulator (`nvidia-fp64` or `qpp-cpu`).
Hardware backends are not supported without further porting work (see CUDA-Q docs on QPU targets).

### QAOA vs VQE

- **QAOA** (`c2q_maxcut`, etc.): Problem-specific cost + mixer Hamiltonian.
  Better theoretical guarantees; depth grows with `layers` (p).
- **VQE** (`c2q_maxcut_vqe`, etc.): Hardware-efficient RY+CZ ansatz.
  Shallower circuit, may miss optimal for strongly constrained problems.

QAOA with `layers=2` is the default and works well for small graphs (≤ 10 nodes).
Increase `layers` for larger or harder instances at the cost of more circuit depth.

## Implementation Notes

### No `cudaq::adjoint`
All inverse circuits (inverse QFT, etc.) are implemented directly as separate kernels.
`cudaq::adjoint` on variable-bound inner loops triggers an MLIR crash in the
`apply-op-specialization` pass (`cloneReversedLoop`). This constraint applies to
all quantum kernels in this library.

### QUBO → Ising conversion
Graph problems are formulated as QUBO matrices, then converted to Ising Hamiltonians
via `x_i = (1 - σ_i)/2`. The conversion is in `include/c2cudaq/internal.h`.

### Factoring approach
`c2q_factor` iterates candidate factor pairs classically (up to √n), then quantum-verifies
each by running the QFT multiplier and checking the measured product. This is not full
Grover search (which would require controlled multiply + uncompute, blocked by the adjoint
constraint), but it uses the quantum arithmetic kernel as the verification oracle.

## File Structure

```
c2cudaq/
  include/
    c2cudaq.h           Public API (no cudaq.h dependency)
    c2cudaq/internal.h  QUBO builders, Ising conversion, decoders (inline)
  src/
    qubo.cpp            Compilation unit for internal.h (empty body)
    arith.cpp           ADD, SUB, MUL quantum kernels
    qaoa.cpp            QAOA + VQE kernels, all graph problem implementations
    grover.cpp          FACTOR (quantum-verified)
  tests/
    test_arith.cpp      ADD/SUB/MUL correctness tests
    test_graph.cpp      MaxCut, MIS, VC, Clique, KColor, TSP tests
    test_factor.cpp     Factor tests (15, 21, 35, 77)
  examples/
    example_arith.cpp   Arithmetic usage examples
    example_graph.cpp   Graph optimization examples with all overloads
    example_factor.cpp  Factoring examples
  nvqpp_wrap.sh         nvq++ wrapper that strips CMake dep-file flags
  CMakeLists.txt
```
