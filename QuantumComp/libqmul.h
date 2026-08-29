#pragma once
#include <cudaq.h>
#include <string>

void print_as_decimal(const cudaq::sample_result& result,
                      const std::string& label);

// Both inputs encoded in n_qubits bits; accumulator is 2*n_qubits bits.
cudaq::sample_result qft_multiply(int a, int b, int n_qubits);

// Inputs encoded with independent bit widths; accumulator is size_a+size_b
// bits.
cudaq::sample_result qft_multiply_optimized(int a,
                                            int b,
                                            int size_a,
                                            int size_b);
