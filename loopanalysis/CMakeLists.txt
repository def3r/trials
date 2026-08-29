cmake_minimum_required(VERSION 3.20)
project(MinPass)

if(NOT DEFINED LLVM_DIR AND DEFINED ENV{LLVM_PROJECT_INSTALL})
  set(LLVM_DIR "$ENV{LLVM_PROJECT_INSTALL}/lib/cmake/llvm")
endif()

find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

add_llvm_pass_plugin(MinPass
  pass.cpp
  maxcut_pass.cpp
  tsp_pass.cpp
  kcolor_pass.cpp
  clique_pass.cpp
  factor_pass.cpp
)

# Testing
enable_testing()
find_package(Python3 COMPONENTS Interpreter)

# Locate lit. Preference order:
#   1. -DLLVM_EXTERNAL_LIT=<path>   user override, persisted in CMakeCache
#   2. llvm-lit in the LLVM install's bin dir (LLVM_TOOLS_BINARY_DIR)
#   3. llvm-lit in a sibling build/ directory — common dev layout where
#      <prefix>/install/ and <prefix>/build/ sit next to each other
if(NOT LLVM_EXTERNAL_LIT)
  get_filename_component(_sibling_build_bin
    "${LLVM_TOOLS_BINARY_DIR}/../../build/bin" ABSOLUTE)
  find_program(LLVM_EXTERNAL_LIT
    NAMES llvm-lit lit.py lit
    HINTS
      "${LLVM_TOOLS_BINARY_DIR}"
      "${_sibling_build_bin}"
    DOC "Path to the lit test runner (llvm-lit binary or lit.py script)"
  )
  unset(_sibling_build_bin)
endif()

if(Python3_EXECUTABLE AND LLVM_EXTERNAL_LIT)
  add_lit_testsuite(check-maxcut "Running maxcut-pass lit tests"
    ${CMAKE_CURRENT_SOURCE_DIR}/test/maxcut
    DEPENDS MinPass
  )
  message(STATUS "check-maxcut target enabled (lit: ${LLVM_EXTERNAL_LIT})")
  add_lit_testsuite(check-tsp "Running tsp-pass lit tests"
    ${CMAKE_CURRENT_SOURCE_DIR}/test/tsp
    DEPENDS MinPass
  )
  message(STATUS "check-tsp target enabled (lit: ${LLVM_EXTERNAL_LIT})")
  add_lit_testsuite(check-kcolor "Running kcolor-pass lit tests"
    ${CMAKE_CURRENT_SOURCE_DIR}/test/kcolor
    DEPENDS MinPass
  )
  message(STATUS "check-kcolor target enabled (lit: ${LLVM_EXTERNAL_LIT})")
  add_lit_testsuite(check-clique "Running clique-pass lit tests"
    ${CMAKE_CURRENT_SOURCE_DIR}/test/clique
    DEPENDS MinPass
  )
  message(STATUS "check-clique target enabled (lit: ${LLVM_EXTERNAL_LIT})")
  add_lit_testsuite(check-factor "Running factor-pass lit tests"
    ${CMAKE_CURRENT_SOURCE_DIR}/test/factor
    DEPENDS MinPass
  )
  message(STATUS "check-factor target enabled (lit: ${LLVM_EXTERNAL_LIT})")
elseif(NOT Python3_EXECUTABLE)
  message(STATUS "check-maxcut disabled — Python 3 interpreter not found")
  message(STATUS "check-tsp disabled — Python 3 interpreter not found")
  message(STATUS "check-kcolor disabled — Python 3 interpreter not found")
  message(STATUS "check-clique disabled — Python 3 interpreter not found")
  message(STATUS "check-factor disabled — Python 3 interpreter not found")
else()
  message(STATUS "check-maxcut disabled — "
    "pass -DLLVM_EXTERNAL_LIT=/path/to/llvm-lit to enable")
  message(STATUS "check-tsp disabled — "
    "pass -DLLVM_EXTERNAL_LIT=/path/to/llvm-lit to enable")
  message(STATUS "check-kcolor disabled — "
    "pass -DLLVM_EXTERNAL_LIT=/path/to/llvm-lit to enable")
  message(STATUS "check-clique disabled — "
    "pass -DLLVM_EXTERNAL_LIT=/path/to/llvm-lit to enable")
  message(STATUS "check-factor disabled — "
    "pass -DLLVM_EXTERNAL_LIT=/path/to/llvm-lit to enable")
endif()
