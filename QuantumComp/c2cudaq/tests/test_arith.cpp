#include <c2cudaq.h>
#include <iostream>
#include <cstdlib>

static int failures = 0;

static void check(const char* name, int64_t got, int64_t expected) {
    if (got == expected) {
        std::cout << "[PASS] " << name << " = " << got << "\n";
    } else {
        std::cout << "[FAIL] " << name << ": got " << got
                  << ", expected " << expected << "\n";
        ++failures;
    }
}

int main() {
    // ADD
    check("c2q_add(3, 5)",   c2q_add(3, 5),   8);
    check("c2q_add(7, 4)",   c2q_add(7, 4),   11);
    check("c2q_add(0, 7)",   c2q_add(0, 7),   7);
    check("c2q_add(12, 15)", c2q_add(12, 15), 27);
    check("c2q_add(100, 55)",c2q_add(100,55), 155);

    // SUB
    check("c2q_sub(8, 3)",   c2q_sub(8, 3),   5);
    check("c2q_sub(11, 4)",  c2q_sub(11, 4),  7);
    check("c2q_sub(15, 15)", c2q_sub(15, 15), 0);
    check("c2q_sub(3, 5)",   c2q_sub(3, 5),   -2);
    check("c2q_sub(100, 55)",c2q_sub(100,55), 45);

    // MUL
    check("c2q_mul(3, 5)",         c2q_mul(3, 5),         15);
    check("c2q_mul(7, 11)",        c2q_mul(7, 11),        77);
    check("c2q_mul(13,11,4,4)",    c2q_mul(13,11,4,4),    143);
    check("c2q_mul(2, 8)",         c2q_mul(2, 8),         16);

    std::cout << "\n" << (failures == 0 ? "All arithmetic tests PASSED."
                                        : std::to_string(failures) + " test(s) FAILED.") << "\n";
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
