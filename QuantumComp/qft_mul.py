import cudaq
import math

# Based on
# 1. https://nvidia.github.io/cuda-quantum/latest/applications/python/quantum_fourier_transform.html
# 2. https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RGQFTMultiplier

# 1. Define a standalone QFT kernel
@cudaq.kernel
def apply_qft(qubits: cudaq.qview):
    M = qubits.size()
    for i in range(M - 1, -1, -1):
        h(qubits[i])
        for j in range(i - 1, -1, -1):
            angle = math.pi / (2 ** (i - j))
            r1.ctrl(angle, [qubits[j]], qubits[i])

# 2. Define the Multiplier Kernel
@cudaq.kernel
def qft_multiply(a: int, b: int, n_qubits: int):
    # Allocators KEEP the cudaq prefix
    reg_a = cudaq.qvector(n_qubits)
    reg_b = cudaq.qvector(n_qubits)
    acc = cudaq.qvector(2 * n_qubits) 

    # --- Step 1: State Preparation ---
    for i in range(n_qubits):
        if (a >> i) & 1:
            x(reg_a[i])

    for j in range(n_qubits):
        if (b >> j) & 1:
            x(reg_b[j])

    # --- Step 2: QFT on Accumulator ---
    apply_qft(acc)

    # --- Step 3: Doubly-Controlled Phase Additions ---
    M = 2 * n_qubits
    for i in range(n_qubits):
        for j in range(n_qubits):
            p = i + j
            for k in range(p, M):
                angle = math.pi / (2 ** (k - p))
                # Passing a list of controls creates a multi-controlled gate.
                r1.ctrl(angle, [reg_a[i], reg_b[j]], acc[k]) 

    # --- Step 4: Inverse QFT ---
    # Algorithmic modifiers KEEP the cudaq prefix
    cudaq.adjoint(apply_qft, acc)

    # --- Step 5: Measurement ---
    mz(acc)


# --- Execution and Helper Function ---
def print_as_decimal(result, label):
    print(f"--- Output for {label} ---")
    for bitstring, count in result.items():
        reversed_bitstring = bitstring[::-1]
        decimal_val = int(reversed_bitstring, 2)
        print(f"Raw String: {bitstring} | Decimal: {decimal_val} | Shots: {count}")
    print()

@cudaq.kernel
def qft_multiply_optimized(a: int, b: int, size_a: int, size_b: int):
    reg_a = cudaq.qvector(size_a)
    reg_b = cudaq.qvector(size_b)

    # The max output size of a*b is size_a + size_b
    acc_size = size_a + size_b
    acc = cudaq.qvector(acc_size) 

    for i in range(size_a):
        if (a >> i) & 1:
            x(reg_a[i])

    for j in range(size_b):
        if (b >> j) & 1:
            x(reg_b[j])

    apply_qft(acc)

    # Doubly-controlled additions mapped to the correct sizes
    for i in range(size_a):
        for j in range(size_b):
            p = i + j
            for k in range(p, acc_size):
                angle = math.pi / (2 ** (k - p))
                r1.ctrl(angle, [reg_a[i], reg_b[j]], acc[k])

    cudaq.adjoint(apply_qft, acc)
    mz(acc)

# Execute: 133 (requires 8 bits), 11 (requires 4 bits)
# Total memory required: ~268 MB.
result_133_11 = cudaq.sample(qft_multiply_optimized, 133, 11, 8, 4)
print_as_decimal(result_133_11, "133 * 11")

# Test 1: Multiply 7 x 11
result_7_11 = cudaq.sample(qft_multiply, 7, 11, 4)
print_as_decimal(result_7_11, "7 * 11")

# Test 2: Multiply 3 x 5
result_3_5 = cudaq.sample(qft_multiply, 3, 5, 3)
print_as_decimal(result_3_5, "3 * 5")
