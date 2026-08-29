#!/usr/bin/env python3
"""
c2q_qasm.py — Standalone QASM exporter for C2|Q> problems.

Supports: ADD, SUB, MUL, Factor, MaxCut, MIS, TSP, Clique, KColor, VC

Each family exports:
  Arithmetic (ADD/SUB/MUL) : <basename>_circuit.qasm
  Factor                   : <basename>_grover.qasm
  Graph (MaxCut/MIS/…)     : <basename>_qaoa.qasm, <basename>_vqe.qasm
  MIS (additionally)       : <basename>_grover.qasm   (requires pysat)

Usage:
  python c2q_qasm.py --input problem.json --output-dir ./out
  python c2q_qasm.py --input problem.json --output-dir ./out --basename my_name

Input JSON (same format as the main C2Q pipeline):
  { "family": "ADD", "instance": { "operands": [13, 5] } }
  { "family": "MaxCut", "instance": { "graph_rep": "edge_list",
                                       "graphs": { "G1": [[0,1],[1,2],[2,0]] } } }

Dependencies: qiskit  numpy  networkx
              pysat   (MIS Grover only — pip install python-sat)
"""

import json
import math
import os
import argparse
import sys

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import (
    VBERippleCarryAdder,
    RGQFTMultiplier,
    TwoLocal,
    PauliEvolutionGate,
)
from qiskit.quantum_info import SparsePauliOp
import qiskit.qasm3


# ─── 1. Bit utilities ─────────────────────────────────────────────────────────

def _dec_to_complement(num, n_bits):
    """Two's-complement binary, returned as a little-endian list of ints."""
    if num >= 0:
        s = bin(num)[2:].zfill(n_bits)
        if len(s) > n_bits:
            raise ValueError(f"{num} does not fit in {n_bits} bits")
    else:
        s = bin((1 << n_bits) + num)[2:]
    return [int(b) for b in reversed(s)]


# ─── 2. QASM output ───────────────────────────────────────────────────────────

def _write_qasm(qc, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(qiskit.qasm3.dumps(qc))
    print(f"  Exported → {path}")


# ─── 3. Graph builder ─────────────────────────────────────────────────────────

def _build_graph(payload, graph_rep=None):
    """
    Build a nx.Graph from an edge list [[u,v], [u,v,w], ...]
    or an adjacency/distance matrix [[w00, w01, ...], ...].
    Pass graph_rep="edge_list" or "matrix" to skip auto-detection.
    """
    G = nx.Graph()

    use_matrix = False
    if graph_rep == "matrix":
        use_matrix = True
    elif graph_rep != "edge_list":
        # Auto-detect: square AND all inner elements are scalars → matrix
        arr = np.array(payload, dtype=object)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            if all(not isinstance(payload[i][j], (list, tuple))
                   for i in range(len(payload))
                   for j in range(len(payload[i]))):
                use_matrix = True

    if use_matrix:
        m = np.array(payload, dtype=float)
        n = m.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if m[i, j] > 0:
                    G.add_edge(i, j, weight=float(m[i, j]))
    else:
        for edge in payload:
            if len(edge) == 2:
                G.add_edge(edge[0], edge[1], weight=1)
            else:
                G.add_edge(edge[0], edge[1], weight=edge[2])

    return G


# ─── 4. QUBO constructors ─────────────────────────────────────────────────────

def _qubo_maxcut(G):
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    Q = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        i, j = idx[u], idx[v]
        Q[i, i] -= w
        Q[j, j] -= w
        Q[i, j] += 2 * w
    return Q


def _qubo_mis(G, B=1.0):
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = 2 * B
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] -= B
    for u, v in G.edges():
        i, j = sorted([idx[u], idx[v]])
        Q[i, j] += A
    return Q


def _qubo_vc(G, B=1.0):
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = 2 * B
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] += B
    for u, v in G.edges():
        i, j = sorted([idx[u], idx[v]])
        Q[i, i] -= A
        Q[j, j] -= A
        Q[i, j] += A
    return Q


def _qubo_clique(G, K=None, B=1.0):
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        raise ValueError("Empty graph")
    if K is None:
        K = max(1, n - 1)
    K = max(1, min(K, n))
    A = K * B + 10
    Q = np.zeros((n, n))
    lc = -2 * A * K + A
    for i in range(n):
        Q[i, i] += lc
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += 2 * A
            if G.has_edge(nodes[i], nodes[j]):
                Q[i, j] -= B
    return Q


def _qubo_kcolor(G, k=3, A=2.0):
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    Q = np.zeros((N * k, N * k))
    for v in nodes:
        vi = idx[v]
        for i in range(k):
            Q[vi * k + i, vi * k + i] -= A
        for i in range(k):
            for j in range(i + 1, k):
                Q[vi * k + i, vi * k + j] += 2 * A
    for u, v in G.edges():
        ui, vi = idx[u], idx[v]
        for i in range(k):
            Q[ui * k + i, vi * k + i] += A
    return Q


def _qubo_tsp(G, B=1.0):
    nodes = list(G.nodes())
    n = len(nodes)
    max_w = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)
    A = B * max_w + 10
    Q = np.zeros((n * n, n * n))
    # distance cost (H_B)
    for v in range(n):
        for u in range(n):
            nv, nu = nodes[v], nodes[u]
            if nv != nu and G.has_edge(nv, nu):
                w = G[nv][nu].get("weight", 1)
                for p in range(n - 1):
                    Q[v * n + p, u * n + (p + 1)] += B * w
    # each position gets exactly one city (H_A)
    for v in range(n):
        for j in range(n):
            Q[v * n + j, v * n + j] -= A
            for k in range(j + 1, n):
                Q[v * n + j, v * n + k] += 2 * A
    # each city visited exactly once (H_A)
    for j in range(n):
        for v in range(n):
            Q[v * n + j, v * n + j] -= A
            for u in range(v + 1, n):
                Q[v * n + j, u * n + j] += 2 * A
    # penalty for non-edges (H_A)
    for v in range(n):
        for u in range(n):
            nv, nu = nodes[v], nodes[u]
            if nv != nu and not G.has_edge(nv, nu):
                for p in range(n - 1):
                    Q[v * n + p, u * n + (p + 1)] += A
    return Q


# ─── 5. QAOA / VQE circuits ───────────────────────────────────────────────────

def _qubo_to_ising(qubo):
    """Convert QUBO matrix to SparsePauliOp Ising Hamiltonian."""
    n = len(qubo)
    ops = []
    offset = 0.0
    for i in range(n):
        for j in range(i, n):
            pauli = ["I"] * n
            if i == j:
                pauli[i] = "Z"
                val = (-0.5 * qubo[i][i]
                       - 0.25 * float(np.sum(qubo[i][i + 1:]))
                       - 0.25 * float(np.sum(qubo[:i, i])))
                offset += 0.5 * qubo[i][i]
            else:
                pauli[i] = "Z"
                pauli[j] = "Z"
                val = 0.25 * qubo[i][j]
                offset += 0.25 * qubo[i][j]
            if val != 0.0:
                ops.append(("".join(pauli), val))
    if not ops:
        ops = [("I" * n, 0.0)]
    return SparsePauliOp.from_list(ops), offset


def _qaoa_circuit(qubo, layers=1):
    """Parametrized QAOA circuit (no optimization). Uses PauliEvolutionGate cost layer."""
    n = len(qubo)
    ising, _ = _qubo_to_ising(qubo)
    params = ParameterVector("theta", 2 * layers)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.barrier()
    for i in range(layers):
        qc.append(PauliEvolutionGate(ising, params[2 * i]), range(n))
        qc.barrier()
        qc.rx(2 * params[2 * i + 1], range(n))
    qc.measure_all()
    return qc


def _vqe_circuit(qubo, layers=1):
    """Hardware-efficient VQE ansatz (TwoLocal RY+CZ). No optimization."""
    n = len(qubo)
    qc = TwoLocal(n, "ry", "cz", insert_barriers=True, reps=layers)
    qc.measure_all()
    return qc


# ─── 6. CNF oracle + Grover (MIS Grover path) ────────────────────────────────

def _mis_cnf(G):
    """Build MIS CNF formula (maximality + independence clauses)."""
    try:
        from pysat.formula import CNF
    except ImportError:
        raise ImportError(
            "pysat is required for MIS Grover export.\n"
            "Install it with: pip install python-sat"
        )
    cnf = CNF()
    nodes = sorted(G.nodes())
    var = lambda v: v + 1  # SAT variable index (1-based)
    for u, v in G.edges():
        cnf.append([-var(u), -var(v)])
    for v in nodes:
        neighbors = list(G.neighbors(v))
        if neighbors:
            cnf.append([var(nb) for nb in neighbors] + [var(v)])
    return cnf


def _cnf_to_circuit(cnf):
    """
    Convert a CNF formula to a quantum circuit that counts satisfied clauses
    and flips a result qubit if all clauses are satisfied.
    Mirrors circuits_library.cnf_to_quantum_circuit_optimized.
    """
    from qiskit.circuit.library import OR
    nv = cnf.nv
    nc = len(cnf.clauses)
    n_count = nc.bit_length()
    total = nv + n_count + 1 + 1  # variables + count ancillas + final_ancilla + output
    qc = QuantumCircuit(total)
    count_ancillas = list(range(nv, nv + n_count))
    final_ancilla = nv + n_count
    final_result = final_ancilla + 1

    for clause in cnf.clauses:
        clause_qubits = []
        negations = []
        for lit in clause:
            qi = abs(lit) - 1
            if lit < 0:
                qc.x(qi)
                negations.append(qi)
            clause_qubits.append(qi)

        or_gate = OR(len(clause_qubits))
        qc.append(or_gate, clause_qubits + [final_ancilla])

        # Increment count ancillas
        for j in range(1, n_count):
            ctrl = [final_ancilla] + count_ancillas[:n_count - j]
            qc.mcx(ctrl, count_ancillas[n_count - j])
        qc.cx(final_ancilla, count_ancillas[0])

        qc.reset(final_ancilla)
        if negations:
            qc.x(negations)
        qc.barrier()

    # Check if count == num_clauses
    ctrl = [count_ancillas[j] for j in range(n_count) if (nc >> j) & 1]
    if ctrl:
        qc.mcx(ctrl, final_ancilla)
    qc.cx(final_ancilla, final_result)
    qc.reset(count_ancillas)
    qc.reset(final_ancilla)
    return qc


def _cnf_to_oracle(cnf):
    """Phase-kickback oracle from CNF. Mirrors cnf_to_quantum_oracle_optimized."""
    inner = _cnf_to_circuit(cnf)
    qc = QuantumCircuit(inner.num_qubits)
    qc.barrier()
    qc.x(inner.num_qubits - 1)
    qc.h(inner.num_qubits - 1)
    qc.barrier()
    qc.compose(inner, inplace=True)
    qc.h(inner.num_qubits - 1)
    qc.x(inner.num_qubits - 1)
    qc.barrier()
    return qc


def _grover_circuit(oracle, state_prep, objective_qubits, working_qubits, iterations=1):
    """Grover's algorithm around a given phase oracle."""
    nq = oracle.num_qubits
    qr = QuantumRegister(nq)
    cr = ClassicalRegister(len(objective_qubits))
    qc = QuantumCircuit(qr, cr)
    qc = qc.compose(state_prep)
    for _ in range(iterations):
        oracle.name = "oracle"
        qc.append(oracle, qr)
        qc.h(working_qubits)
        qc.x(working_qubits)
        qc.h(working_qubits[-1])
        qc.mcx(working_qubits[:-1], working_qubits[-1])
        qc.h(working_qubits[-1])
        qc.x(working_qubits)
        qc.h(working_qubits)
    qc.global_phase = math.pi
    qc.measure(objective_qubits, cr)
    return qc


def _mis_grover(G, iterations=1):
    """Build the full Grover circuit for the MIS problem."""
    cnf = _mis_cnf(G)
    oracle = _cnf_to_oracle(cnf)
    n_nodes = G.number_of_nodes()
    state_prep = QuantumCircuit(oracle.num_qubits)
    state_prep.h(list(range(n_nodes)))
    obj = list(range(n_nodes))
    return _grover_circuit(oracle, state_prep, obj, obj, iterations)


# ─── 7. Factor oracle ─────────────────────────────────────────────────────────

def _factor_oracle(n):
    """
    Build the Grover oracle for factorization of n.
    Returns (oracle_circuit, prep_state, obj_bits, working_bits).
    Mirrors circuits_library.quantum_factor_mul_oracle.
    """
    num_result = n.bit_length()
    num_state = math.ceil(num_result / 2)
    obj_bits = list(range(num_state))
    q = QuantumRegister(num_state * 2 + num_result + 1, "q")
    circuit = QuantumCircuit(q)
    prep_state = QuantumCircuit(q)
    working_bits = list(range(num_state * 2))
    prep_state.h(working_bits)

    mul = RGQFTMultiplier(num_state_qubits=num_state, num_result_qubits=num_result)
    circuit.x(q[circuit.num_qubits - 1])
    circuit.h(q[circuit.num_qubits - 1])
    circuit = circuit.compose(mul)

    binary_n = format(n, f"0{num_result}b")
    for i in range(num_result):
        if binary_n[num_result - i - 1] == "0":
            circuit.x(q[i + num_state * 2])
    circuit.mcx(list(range(num_state * 2, num_state * 2 + num_result)), q[circuit.num_qubits - 1])
    for i in range(num_result):
        if binary_n[num_result - i - 1] == "0":
            circuit.x(q[i + num_state * 2])

    circuit = circuit.compose(mul.inverse())
    return circuit, prep_state, obj_bits, working_bits


def _factor_grover(n, iterations=2):
    oracle, prep, obj_bits, working_bits = _factor_oracle(n)
    return _grover_circuit(oracle, prep, obj_bits, working_bits, iterations)


# ─── 8. Arithmetic circuits ───────────────────────────────────────────────────

def _add_circuit(left, right):
    n_bits = max(left.bit_length(), right.bit_length()) + 1
    left_l = _dec_to_complement(left, n_bits)
    right_l = _dec_to_complement(right, n_bits)
    if left * right > 0:
        qc = QuantumCircuit(3 * n_bits + 1, n_bits + 1)
    else:
        qc = QuantumCircuit(3 * n_bits + 1, n_bits)
    for i in range(n_bits):
        if left_l[i] == 1:
            qc.x(i)
        if right_l[i] == 1:
            qc.x(n_bits + i)
    for i in range(n_bits):
        qc.ccx(i, n_bits + i, 2 * n_bits + i + 1)
        qc.cx(i, n_bits + i)
        qc.ccx(n_bits + i, 2 * n_bits + i, 2 * n_bits + i + 1)
        qc.cx(n_bits + i, 2 * n_bits + i)
        qc.cx(i, n_bits + i)
    for i in range(n_bits):
        qc.measure(i + 2 * n_bits, i)
    if left * right > 0:
        qc.measure(3 * n_bits, n_bits)
    return qc


def _sub_circuit(left, right):
    n_bits = max(left.bit_length(), right.bit_length())
    minuend = _dec_to_complement(left, n_bits)
    subtrahend = _dec_to_complement(right, n_bits)
    adder = VBERippleCarryAdder(num_state_qubits=n_bits)
    nq = len(adder.qubits)
    if left * right < 0:
        qc = QuantumCircuit(nq, n_bits + 1)
    else:
        qc = QuantumCircuit(nq, n_bits)
    for i in range(n_bits):
        if minuend[i] == 1:
            qc.x(i + 1)
        if subtrahend[i] == 1:
            qc.x(i + 1 + n_bits)
    qc.barrier()
    qc.x(range(n_bits + 1, 2 * n_bits + 1))
    qc.x(0)
    qc.append(adder, range(nq))
    for i in range(n_bits):
        qc.measure(i + n_bits + 1, i)
    if left * right < 0:
        qc.measure(n_bits + n_bits + 1, n_bits)
    return qc


def _mul_circuit(left, right):
    num_state = max(left.bit_length(), right.bit_length())
    num_result = num_state * 2
    q = QuantumRegister(num_state * 2 + num_result, "q")
    c = ClassicalRegister(num_result, "c")
    qc = QuantumCircuit(q, c)
    for i in range(left.bit_length()):
        if (left >> i) & 1:
            qc.x(q[i])
    for i in range(right.bit_length()):
        if (right >> i) & 1:
            qc.x(q[i + num_state])
    mul = RGQFTMultiplier(num_state_qubits=num_state, num_result_qubits=num_result)
    qc = qc.compose(mul)
    for i in range(num_result):
        qc.measure(q[i + num_state * 2], c[i])
    return qc


# ─── 9. Per-family QASM export ────────────────────────────────────────────────

def _export_arithmetic(family, left, right, out_dir, basename):
    if family == "ADD":
        qc = _add_circuit(left, right)
    elif family == "SUB":
        qc = _sub_circuit(left, right)
    elif family == "MUL":
        qc = _mul_circuit(left, right)
    else:
        raise ValueError(f"Unknown arithmetic family: {family}")
    path = os.path.join(out_dir, f"{basename}_circuit.qasm")
    _write_qasm(qc.decompose(reps=10), path)
    return {"circuit": path}


def _export_factor(n, out_dir, basename):
    qc = _factor_grover(n)
    path = os.path.join(out_dir, f"{basename}_grover.qasm")
    _write_qasm(qc.decompose(reps=10), path)
    return {"grover": path}


def _export_graph(family, G, params, out_dir, basename):
    """Export QAOA + VQE (and Grover for MIS) for a graph problem."""
    if family == "MaxCut":
        qubo = _qubo_maxcut(G)
    elif family == "MIS":
        qubo = _qubo_mis(G)
    elif family == "VC":
        qubo = _qubo_vc(G)
    elif family == "Clique":
        K = params.get("k") or params.get("K")
        qubo = _qubo_clique(G, K=K)
    elif family == "KColor":
        k = params.get("k") or params.get("colors") or 3
        qubo = _qubo_kcolor(G, k=int(k))
    elif family == "TSP":
        qubo = _qubo_tsp(G)
    else:
        raise ValueError(f"Unknown graph family: {family}")

    paths = {}

    # QAOA
    try:
        qc = _qaoa_circuit(qubo, layers=1)
        path = os.path.join(out_dir, f"{basename}_qaoa.qasm")
        _write_qasm(qc.decompose(reps=10), path)
        paths["qaoa"] = path
    except Exception as e:
        print(f"  Warning: QAOA export failed: {e}")

    # VQE
    try:
        qc = _vqe_circuit(qubo, layers=1)
        path = os.path.join(out_dir, f"{basename}_vqe.qasm")
        _write_qasm(qc.decompose(reps=10), path)
        paths["vqe"] = path
    except Exception as e:
        print(f"  Warning: VQE export failed: {e}")

    # Grover (MIS only)
    if family == "MIS":
        try:
            qc = _mis_grover(G)
            path = os.path.join(out_dir, f"{basename}_grover.qasm")
            _write_qasm(qc.decompose(reps=10), path)
            paths["grover"] = path
        except ImportError as e:
            print(f"  Warning: Grover skipped — {e}")
        except Exception as e:
            print(f"  Warning: Grover export failed: {e}")

    return paths


# ─── 10. JSON parsing ─────────────────────────────────────────────────────────

_FAMILY_ALIASES = {
    "add": "ADD", "addition": "ADD",
    "sub": "SUB", "subtract": "SUB", "subtraction": "SUB",
    "mul": "MUL", "multiply": "MUL", "multiplication": "MUL",
    "factor": "Factor", "factorization": "Factor", "factorisation": "Factor",
    "maxcut": "MaxCut", "max_cut": "MaxCut", "max-cut": "MaxCut",
    "mis": "MIS", "maximumindependentset": "MIS", "independentset": "MIS",
    "tsp": "TSP", "travellingsalesman": "TSP", "travelingsalesman": "TSP",
    "clique": "Clique",
    "kcolor": "KColor", "k_color": "KColor", "graphcoloring": "KColor",
    "vc": "VC", "vertexcover": "VC", "minimumvertexcover": "VC",
}

_GRAPH_FAMILIES = {"MaxCut", "MIS", "TSP", "Clique", "KColor", "VC"}
_ARITH_FAMILIES = {"ADD", "SUB", "MUL"}


def _canon_family(raw):
    if raw is None:
        return None
    key = raw.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return _FAMILY_ALIASES.get(key, raw.strip())


def _parse_input(path):
    with open(path) as f:
        task = json.load(f)

    raw_family = task.get("family") or task.get("problem_type")
    family = _canon_family(raw_family)
    instance = task.get("instance") or task.get("data") or task.get("json") or {}
    params = task.get("parameters") or task.get("config") or {}

    if family is None:
        raise ValueError("JSON must include 'family' or 'problem_type'.")

    if family in _ARITH_FAMILIES or family == "Factor":
        # Normalise operands
        if isinstance(instance, (list, tuple)) and len(instance) == 2:
            operands = list(instance)
        elif "operands" in instance:
            operands = instance["operands"]
        elif "a" in instance and "b" in instance:
            operands = [instance["a"], instance["b"]]
        else:
            operands = None

        if family == "Factor":
            n_val = instance.get("n") or instance.get("value") or instance.get("number")
            if n_val is None:
                raise ValueError("Factor instance must include 'n', 'value', or 'number'.")
            return family, int(n_val), params

        if operands is None or len(operands) != 2:
            raise ValueError(f"{family} instance must include 'operands', or 'a'/'b'.")
        return family, (int(operands[0]), int(operands[1])), params

    if family in _GRAPH_FAMILIES:
        # Extract graph payload
        if "graphs" in instance:
            payload = next(iter(instance["graphs"].values()))
        elif "graph" in instance:
            payload = instance["graph"]
        elif "edges" in instance:
            payload = instance["edges"]
        else:
            payload = instance

        graph_rep = instance.get("graph_rep")
        G = _build_graph(payload, graph_rep=graph_rep)
        return family, G, params

    raise ValueError(
        f"Unknown family {family!r}. "
        f"Supported: ADD, SUB, MUL, Factor, MaxCut, MIS, TSP, Clique, KColor, VC"
    )


# ─── 11. Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Export quantum circuits as QASM 3 files from a C2Q JSON problem description."
    )
    ap.add_argument("--input", required=True, help="Path to the JSON problem file.")
    ap.add_argument("--output-dir", default=".", help="Directory to write .qasm files into.")
    ap.add_argument(
        "--basename",
        default=None,
        help="Base name for output files (default: input filename stem).",
    )
    args = ap.parse_args()

    basename = args.basename or os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir

    print(f"Loading {args.input} ...")
    family, data, params = _parse_input(args.input)
    print(f"Family: {family}")

    os.makedirs(out_dir, exist_ok=True)

    if family in _ARITH_FAMILIES:
        left, right = data
        print(f"Operands: {left}, {right}")
        paths = _export_arithmetic(family, left, right, out_dir, basename)

    elif family == "Factor":
        n_val = data
        print(f"Factoring: {n_val}")
        paths = _export_factor(n_val, out_dir, basename)

    elif family in _GRAPH_FAMILIES:
        G = data
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        paths = _export_graph(family, G, params, out_dir, basename)

    else:
        print(f"Error: unsupported family {family!r}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {len(paths)} file(s) written to {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
