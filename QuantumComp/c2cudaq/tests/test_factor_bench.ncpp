#include <c2cudaq.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct BenchRow {
    int64_t n;
    // Grover
    int64_t grover_p, grover_q;
    bool    grover_ok;
    double  grover_ms;
    // Shor
    int64_t shor_p, shor_q;
    bool    shor_ok;
    double  shor_ms;
};

static bool valid(int64_t n, int64_t p, int64_t q) {
    bool trivial = (p == 1 && q == n) || (p == n && q == 1);
    return (p * q == n) && !trivial;
}

static BenchRow bench(int64_t n) {
    BenchRow row{};
    row.n = n;

    // Grover
    auto t0 = Clock::now();
    try {
        auto [p, q] = c2q_factor(n);
        row.grover_p  = p;
        row.grover_q  = q;
        row.grover_ok = valid(n, p, q);
    } catch (const std::exception& e) {
        std::cerr << "  Grover threw for n=" << n << ": " << e.what() << "\n";
        row.grover_p = 1; row.grover_q = n; row.grover_ok = false;
    }
    row.grover_ms = Ms(Clock::now() - t0).count();

    // Shor
    t0 = Clock::now();
    try {
        auto [p, q] = c2q_factor_shor(n);
        row.shor_p  = p;
        row.shor_q  = q;
        row.shor_ok = valid(n, p, q);
    } catch (const std::exception& e) {
        std::cerr << "  Shor threw for n=" << n << ": " << e.what() << "\n";
        row.shor_p = 1; row.shor_q = n; row.shor_ok = false;
    }
    row.shor_ms = Ms(Clock::now() - t0).count();

    return row;
}

int main() {
    // Small semiprimes within qubit budget for both algorithms.
    // Grover limit: n ≤ 127.   Shor limit: n ≤ 255.
    std::vector<int64_t> inputs = {15, 21, 35, 49, 77};

    std::cout << "=== Grover vs Shor Factoring Benchmark ===\n\n";
    std::cout << std::left
              << std::setw(6)  << "n"
              << std::setw(22) << "Grover result"
              << std::setw(10) << "Grov ms"
              << std::setw(22) << "Shor result"
              << std::setw(10) << "Shor ms"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    int failures = 0;
    for (int64_t n : inputs) {
        std::cout << std::setw(6) << n << std::flush;
        auto row = bench(n);

        std::string grv = std::to_string(row.grover_p) + "*" +
                          std::to_string(row.grover_q) +
                          (row.grover_ok ? " [ok]" : " [FAIL]");
        std::string shr = std::to_string(row.shor_p) + "*" +
                          std::to_string(row.shor_q) +
                          (row.shor_ok ? " [ok]" : " [FAIL]");

        std::cout << std::setw(22) << grv
                  << std::setw(10) << std::fixed << std::setprecision(1) << row.grover_ms
                  << std::setw(22) << shr
                  << std::setw(10) << row.shor_ms
                  << "\n";

        if (!row.grover_ok) ++failures;
        if (!row.shor_ok)   ++failures;
    }

    std::cout << "\n" << (failures == 0 ? "All benchmark tests PASSED."
                                        : std::to_string(failures) + " test(s) FAILED.")
              << "\n";
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
