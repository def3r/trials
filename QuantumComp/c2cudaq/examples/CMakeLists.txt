function(add_c2_example name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE c2cudaq)
endfunction()

add_c2_example(example_arith  example_arith.cpp)
add_c2_example(example_graph  example_graph.cpp)
add_c2_example(example_factor example_factor.cpp)
