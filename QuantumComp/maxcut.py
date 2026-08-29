import numpy as np
import cudaq
from cudaq import spin
from typing import List

if cudaq.num_available_gpus() > 0 and cudaq.has_target("nvidia"):
    cudaq.set_target("nvidia")
else:
    print("CUDA or GPU support is unavailable. Running with CPU simulator.
          Performance may be significantly reduced.")
    cudaq.set_target("qpp-cpu")

# We'll use the graph below to illustrate how QAOA can be used to
# solve a max cut problem

#       v1  0--------------0 v2
#           |              | \
#           |              |  \
#           |              |   \
#           |              |    \
#       v0  0--------------0 v3-- 0 v4
# The max cut solutions are 01011, 10100, 01010, 10101 .

# First we define the graph nodes (i.e., vertices) and edges as lists of
# integers so that they can be broadcast into a cudaq.kernel.
nodes: List[int] = [0, 1, 2, 3, 4]
edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [3, 4]]
edges_src: List[int] = [edges[i][0] for i in range(len(edges))]
edges_tgt: List[int] = [edges[i][1] for i in range(len(edges))]

# Problem parameters
# The number of qubits we'll need is the same as the number of vertices in our
# graph
qubit_count: int = len(nodes)

# We can set the layer count to be any positive integer.  Larger values will
# create deeper circuits
layer_count: int = 2

# Each layer of the QAOA kernel contains 2 parameters
parameter_count: int = 2 * layer_count

@cudaq.kernel
def qaoaProblem(qubit_0: cudaq.qubit, qubit_1: cudaq.qubit, alpha: float):
    """Build the QAOA gate sequence between two qubits that represent an edge
    of the graph
    Parameters
    ----------
    qubit_0: cudaq.qubit
        Qubit representing the first vertex of an edge
    qubit_1: cudaq.qubit
        Qubit representing the second vertex of an edge
    thetas: List[float]
        Free variable

    Returns
    -------
    cudaq.Kernel
        Subcircuit of the problem kernel for Max-Cut of the graph with a given
        edge
    """
    x.ctrl(qubit_0, qubit_1)
    rz(2.0 * alpha, qubit_1)
    x.ctrl(qubit_0, qubit_1)


# We now define the kernel_qaoa function which will be the QAOA circuit for our
# graph
# Since the QAOA circuit for max cut depends on the structure of the graph,
# we'll feed in global concrete variable values into the kernel_qaoa function
# for the qubit_count, layer_count, edges_src, edges_tgt.
# The types for these variables are restricted to Quake Values (e.g. qubit,
# int, List[int], ...)
# The thetas plaeholder will be our free parameters
@cudaq.kernel
def kernel_qaoa(qubit_count: int, layer_count: int, edges_src: List[int],
                edges_tgt: List[int], thetas: List[float]):
    """Build the QAOA circuit for max cut of the graph with given edges and nodes
    Parameters
    ----------
    qubit_count: int
        Number of qubits in the circuit, which is the same as the number of
        nodes in our graph
    layer_count : int
        Number of layers in the QAOA kernel
    edges_src: List[int]
        List of the first (source) node listed in each edge of the graph, when
        the edges of the graph are listed as pairs of nodes
    edges_tgt: List[int]
        List of the second (target) node listed in each edge of the graph, when
        the edges of the graph are listed as pairs of nodes
    thetas: List[float]
        Free variables to be optimized

    Returns
    -------
    cudaq.Kernel
        QAOA circuit for Max-Cut for max cut of the graph with given edges and
        nodes
    """
    # Let's allocate the qubits
    qreg = cudaq.qvector(qubit_count)
    # And then place the qubits in superposition
    h(qreg)

    # Each layer has two components: the problem kernel and the mixer
    for i in range(layer_count):
        # Add the problem kernel to each layer
        for edge in range(len(edges_src)):
            qubitu = edges_src[edge]
            qubitv = edges_tgt[edge]
            qaoaProblem(qreg[qubitu], qreg[qubitv], thetas[i])
        # Add the mixer kernel to each layer
        for j in range(qubit_count):
            rx(2.0 * thetas[i + layer_count], qreg[j])

# The problem Hamiltonian
# Define a function to generate the Hamiltonian for a max cut problem using the
# graph with the given edges


def hamiltonian_max_cut(edges_src, edges_tgt):
    """Hamiltonian for finding the max cut for the graph with given edges and
    nodes

    Parameters
    ----------
    edges_src: List[int]
        List of the first (source) node listed in each edge of the graph, when
        the edges of the graph are listed as pairs of nodes
    edges_tgt: List[int]
        List of the second (target) node listed in each edge of the graph, when
        the edges of the graph are listed as pairs of nodes

    Returns
    -------
    cudaq.SpinOperator
        Hamiltonian for finding the max cut of the graph with given edges
    """

    hamiltonian = 0

    for edge in range(len(edges_src)):

        qubitu = edges_src[edge]
        qubitv = edges_tgt[edge]
        # Add a term to the Hamiltonian for the edge (u,v)
        hamiltonian += 0.5 * (spin.z(qubitu) * spin.z(qubitv) -
                              spin.i(qubitu) * spin.i(qubitv))

    return hamiltonian

# Specify the optimizer and its initial parameters.
cudaq.set_random_seed(13)
optimizer = cudaq.optimizers.NelderMead()
np.random.seed(13)
optimizer.initial_parameters = np.random.uniform(-np.pi / 8, np.pi / 8,
                                                 parameter_count)
print("Initial parameters = ", optimizer.initial_parameters)

# Generate the Hamiltonian for our graph
hamiltonian = hamiltonian_max_cut(edges_src, edges_tgt)
print(hamiltonian)

# Define the objective, return `<state(params) | H | state(params)>`
# Note that in the `observe` call we list the kernel, the hamiltonian, and then
# the concrete global variable values of our kernel followed by the parameters
# to be optimized.


def objective(parameters):
    return cudaq.observe(kernel_qaoa, hamiltonian, qubit_count, layer_count,
                         edges_src, edges_tgt, parameters).expectation()


# Optimize!
optimal_expectation, optimal_parameters = optimizer.optimize(
    dimensions=parameter_count, function=objective)

# Alternatively we can use the vqe call (just comment out the above code and
# uncomment the code below)
# optimal_expectation, optimal_parameters = cudaq.vqe(
#    kernel=kernel_qaoa,
#    spin_operator=hamiltonian,
#    argument_mapper=lambda parameter_vector: (qubit_count, layer_count,
#       edges_src, edges_tgt, parameter_vector),
#    optimizer=optimizer,
#    parameter_count=parameter_count)

print('optimal_expectation =', optimal_expectation)
print('Therefore, the max cut value is at least ', -1 * optimal_expectation)
print('optimal_parameters =', optimal_parameters)

# Sample the circuit using the optimized parameters
# Since our kernel has more than one argument, we need to list the values for
# each of these variables in order in the `sample` call.
counts = cudaq.sample(kernel_qaoa, qubit_count, layer_count, edges_src,
                      edges_tgt, optimal_parameters)
print(counts)

# Identify the most likely outcome from the sample
max_cut = max(counts, key=lambda x: counts[x])
print('The max cut is given by the partition: ',
      max(counts, key=lambda x: counts[x]))
