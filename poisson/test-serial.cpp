/**
 * @file test-serial.cpp
 * @brief Boost.Test suite for the serial Jacobi Poisson solver.
 *
 * Executes the three verification cases (1-3), which have known analytical
 * solutions, and checks that:
 *   - the Jacobi iteration converges within the maximum iterations
 *   - the residual falls below epsilon
 *   - the maxiumum absolute error across all points against the exact
 *     solution is within a specified tolerance.
 */

#define BOOST_TEST_MODULE PoissonSerialTests
#include <boost/test/included/unit_test.hpp>

#include "test-case.hpp"
#include "solver.hpp"

#include <cmath>
#include <stdexcept>

#define F77NAME(x) x##_
extern "C" {
    void F77NAME(dcopy)(const int& N,
                        const double* x, const int& incx,
                        double* y, const int& incy);
}


namespace {

/**
 * @brief Return the exact analytical solution for a given verification case.
 *
 * @param test_case Verification case identifier (1-3).
 * @param x         x-coordinate.
 * @param y         y-coordinate.
 * @param z         z-coordinate.
 * @return double   Exact solution value.
 * @throws std::invalid_argument if test_case is not 1-3.
 */
double exact_solution(int test_case, double x, double y, double z) {
    switch (test_case) {
        case 1:
            return x * x + y * y + z * z;
            
        case 2:
            return std::sin(M_PI * x) *
                   std::sin(M_PI * y) *
                   std::sin(M_PI * z);
                   
        case 3:
            return std::sin(M_PI * x) *
                   std::sin(4.0 * M_PI * y) *
                   std::sin(8.0 * M_PI * z);
                   
        default:
            throw std::invalid_argument("No exact solution for this test case");
    }
}


/**
 * @brief Set the default grid dimensions for each verification case.
 *
 * Grid sizes follow the brief specification:
 *   - Case 1: 32 x 32 x 32
 *   - Case 2: 64 x 64 x 64
 *   - Case 3: 32 x 64 x 128
 *
 * @param test_case     Verification case identifier (1-3).
 * @param[out] Nx       Grid points in x-direction.
 * @param[out] Ny       Grid points in y-direction.
 * @param[out] Nz       Grid points in z-direction.
 * @throws std::invalid_argument if test_case is not 1-3.
 */
void default_grid(int test_case, int& Nx, int& Ny, int& Nz) {
    switch (test_case) {
        case 1: 
            Nx = 32; Ny = 32;  Nz = 32;  
            break;

        case 2: 
            Nx = 64; Ny = 64;  Nz = 64;  
            break;

        case 3: 
            Nx = 32; Ny = 64;  Nz = 128; 
            break;

        default:
            throw std::invalid_argument("No default grid for this test case");
    }
}


/**
 * @brief Holds the results of a solver run for inspection by tests.
 */
struct SolverResult {
    double* u;          ///< Converged solution array
    int     Nx, Ny, Nz; ///< Grid dimensions used
    double  hx, hy, hz; ///< Grid spacings
    double  residual;   ///< Final residual L2 norm
    int     iter;       ///< Number of iterations performed
    int     max_iter;   ///< Maximum iterations allowed
};


/**
 * @brief Run the Jacobi solver to convergence for a verification case.
 *
 * Allocates all working arrays, initialises the test case, iterates
 * until convergence or the iteration limit, and returns the result.
 *
 * @param test_case Verification case identifier (1-3).
 * @return SolverResult containing the converged solution and diagnostics.
 */
SolverResult run_solver(int test_case) {
    SolverResult res;
    default_grid(test_case, res.Nx, res.Ny, res.Nz);

    int N = res.Nx * res.Ny * res.Nz;

    res.hx   = 1.0 / (res.Nx - 1.0);
    res.hy   = 1.0 / (res.Ny - 1.0);
    res.hz   = 1.0 / (res.Nz - 1.0);

    double a = 1.0 / (res.hx * res.hx);
    double b = 1.0 / (res.hy * res.hy);
    double c = 1.0 / (res.hz * res.hz);

    res.u          = new double[N]();
    double* u_new  = new double[N]();
    double* f      = new double[N]();

    initialise_test_case(test_case, res.u, u_new, f,
                         res.Nx, res.Ny, res.Nz,
                         res.hx, res.hy, res.hz);

    double epsilon = 1e-8;
    double denom   = 1.0 / (2.0 * (a + b + c));

    res.residual   = 1.0;
    res.iter       = 0;
    res.max_iter   = 100000;

    while (res.residual > epsilon && res.iter < res.max_iter) {
        jacobi(res.u, u_new, f, res.Nx, res.Ny, res.Nz, a, b, c, denom);
        F77NAME(dcopy)(N, u_new, 1, res.u, 1);
        res.residual = residual(res.u, f, res.Nx, res.Ny, res.Nz, a, b, c);
        ++res.iter;
    }

    delete[] u_new;
    delete[] f;

    return res;
}


/**
 * @brief Compute the L-infinity error between numerical and exact solutions.
 *
 * @param result SolverResult containing the numerical solution.
 * @param test_case Verification case identifier (1-3).
 * @return double Maximum absolute error over all grid points.
 */
double max_error(const SolverResult& result, int test_case) {
    double max_err = 0.0;

    for (int k = 0; k < result.Nz; ++k) {
        for (int j = 0; j < result.Ny; ++j) {
            for (int i = 0; i < result.Nx; ++i) {
                int idx = i + result.Nx * (j + result.Ny * k);

                double x = i * result.hx;
                double y = j * result.hy;
                double z = k * result.hz;

                double err = std::fabs(result.u[idx] - exact_solution(test_case, x, y, z));
                if (err > max_err) {
                    max_err = err;
                }
            }
        }
    }
    return max_err;
}

} // anonymous namespace


/// Verification case 1
BOOST_AUTO_TEST_CASE(verification_case_1_convergence) {
    SolverResult res = run_solver(1);

    BOOST_TEST_MESSAGE("Case 1: iterations = " << res.iter << ", residual = " << res.residual);

    BOOST_CHECK_MESSAGE(res.iter < res.max_iter,
                        "Solver did not converge within "
                        << res.max_iter << " iterations");
    BOOST_CHECK_LT(res.residual, 1e-8);

    delete[] res.u;
}

BOOST_AUTO_TEST_CASE(verification_case_1_accuracy) {
    SolverResult res = run_solver(1);
    double err = max_error(res, 1);

    BOOST_TEST_MESSAGE("Case 1: Max error = " << err);
    BOOST_CHECK_LT(err, 1e-4);

    delete[] res.u;
}


/// Verification case 2
BOOST_AUTO_TEST_CASE(verification_case_2_convergence) {
    SolverResult res = run_solver(2);

    BOOST_TEST_MESSAGE("Case 2: iterations = " << res.iter
                       << ", residual = " << res.residual);

    BOOST_CHECK_MESSAGE(res.iter < res.max_iter,
                        "Solver did not converge within "
                        << res.max_iter << " iterations");
    BOOST_CHECK_LT(res.residual, 1e-8);

    delete[] res.u;
}

BOOST_AUTO_TEST_CASE(verification_case_2_accuracy) {
    SolverResult res = run_solver(2);
    double err = max_error(res, 2);

    BOOST_TEST_MESSAGE("Case 2: Max error = " << err);
    BOOST_CHECK_LT(err, 1e-3);

    delete[] res.u;
}


/// Verification case 3
BOOST_AUTO_TEST_CASE(verification_case_3_convergence) {
    SolverResult res = run_solver(3);

    BOOST_TEST_MESSAGE("Case 3: iterations = " << res.iter
                       << ", residual = " << res.residual);

    BOOST_CHECK_MESSAGE(res.iter < res.max_iter,
                        "Solver did not converge within "
                        << res.max_iter << " iterations");
    BOOST_CHECK_LT(res.residual, 1e-8);

    delete[] res.u;
}

BOOST_AUTO_TEST_CASE(verification_case_3_accuracy) {
    SolverResult res = run_solver(3);
    double err = max_error(res, 3);

    BOOST_TEST_MESSAGE("Case 3: Max error = " << err);
    BOOST_CHECK_LT(err, 1e-2);

    delete[] res.u;
}
