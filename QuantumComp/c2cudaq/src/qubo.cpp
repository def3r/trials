// Pure C++ - QUBO builders and Ising conversion live in internal.h as inlines.
// This translation unit exists so CMake has a source file for the c2cudaq target
// that does not require quantum compilation.  All heavy logic is in the header.
#include <c2cudaq/internal.h>
