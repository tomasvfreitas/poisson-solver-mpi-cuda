/**
 * @mainpage Poisson Solver Documentation
 *
 * This project implements a serial Jacobi solver for the
 * three-dimensional Poisson equation.
 *
 * The solver uses a finite-difference discretisation and
 * iterates until the L2 norm of the residual falls below
 * a specified tolerance.
 *
 * See the file documentation for implementation details.
 */


/**
 * @file main-serial.cpp
 * @brief Serial Jacobi solver for the 3D Poisson equation.
 *
 * This program solves the Poisson equation \f$ \nabla^2 u = f \f$ on 
 * a three-dimensional grid using Jacobi iteration with Dirichlet boundary conditions.
 *
 * The solution is iteratively updated using a finite-difference discretisation 
 * of the Laplacian operator until the L2 norm of the residual \f$ \|r\|_2 \f$
 * falls below a specified tolerance.
 *
 * The implementation uses the BLAS routine 'dcopy' to update the
 * solution array after each Jacobi iteration.
 *
 * @author Tomas Freitas
 * @date 2026
 */

#include <iostream>
#include <cmath>
#include <string>
#include <boost/program_options.hpp>

#include "solver.hpp"
#include "test-case.hpp"

namespace po = boost::program_options;

#define F77NAME(x) x##_
extern "C" {
    void F77NAME(dcopy)(const int& N,
                        const double* x, const int& incx,
                        double* y, const int& incy);
}

/**
 * @brief Main driver for the serial Jacobi Poisson solver.
 *
 * Sets up the computational grid, initialises the verification problem,
 * performs Jacobi iterations until convergence, and prints the final
 * residual value.
 *
 * @return int Exit status.
 */
int main(int argc, char* argv[]) {
    
    po::options_description opts("Allowed options");
    opts.add_options()
        ("help", "Print available options.")
        ("forcing", po::value<std::string>(), "Input forcing file")
        ("test", po::value<int>()->default_value(1), "Test case to use (1-5)")
        ("Nx", po::value<int>()->default_value(32), "Number of grid points (x)")
        ("Ny", po::value<int>()->default_value(32), "Number of grid points (y)")
        ("Nz", po::value<int>()->default_value(32), "Number of grid points (z)")
        ("epsilon", po::value<double>()->default_value(1e-8), "Residual threshold");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);
    
    /// If user provides option help, print options and exit
    if (vm.count("help")) {
        std::cout << opts << std::endl;
        return 0;
    }
    
    /// Check if user input --test and/or --forcing file
    bool has_test    = vm.count("test") && !vm["test"].defaulted();
    bool has_forcing = vm.count("forcing");
    int  test_case   = vm["test"].as<int>();
    std::string forcing_file;

    /// Check if neither or both were selected and give error
    if (has_test == has_forcing) {
        std::cerr << "Error: Exactly one of --test or --forcing must be provided." << std::endl;
        return 1;
    }
    
    /// Check if given a forcing function the user also specified grid points and give error
    bool Nx_given = vm.count("Nx") && !vm["Nx"].defaulted();
    bool Ny_given = vm.count("Ny") && !vm["Ny"].defaulted();
    bool Nz_given = vm.count("Nz") && !vm["Nz"].defaulted();
    if (has_forcing && (Nx_given || Ny_given || Nz_given)) {
        std::cerr << "Error: --Nx, --Ny and --Nz may only be specified with --test." << std::endl;
        return 1;
    }
    
    /// Checking for valid test number
    if (has_test && (test_case < 1 || test_case > 5)) {
        std::cerr << "Error: test case must be between 1 and 5." << std::endl;
        return 1;
    }

    int Nx = vm["Nx"].as<int>();        ///< Number of grid points in x-direction
    int Ny = vm["Ny"].as<int>();        ///< Number of grid points in y-direction
    int Nz = vm["Nz"].as<int>();        ///< Number of grid points in z-direction
    
    int N  = Nx * Ny * Nz;              ///< Nx by Ny by Nz dimensions for simplicity

    double hx = 1.0 / (Nx - 1.0);       ///< Grid spacing in x-direction
    double hy = 1.0 / (Ny - 1.0);       ///< Grid spacing in y-direction
    double hz = 1.0 / (Nz - 1.0);       ///< Grid spacing in z-direction

    double a = 1.0 / (hx * hx);         ///< Finite-difference coefficient in x-direction
    double b = 1.0 / (hy * hy);         ///< Finite-difference coefficient in y-direction
    double c = 1.0 / (hz * hz);         ///< Finite-difference coefficient in z-direction

    double* u     = new double[N]();    ///< Numerical solution
    double* u_new = new double[N]();    ///< Temporary array to update solution
    double* f     = new double[N]();    ///< Forcing function
    
    
    /// Enforcing Dirichlet boundary conditions and initialising interior values for selected test case
    if (has_test) {
        initialise_test_case(test_case, u, u_new, f, Nx, Ny, Nz, hx, hy, hz);
    }
    
    /// Assigns forcing file
    if (has_forcing) {
        forcing_file = vm["forcing"].as<std::string>();
    }
    
    double epsilon       = vm["epsilon"].as<double>();  ///< Tolerance
    double denom         = 1.0 / (2.0 * (a + b + c));   ///< Setting denom for simplicity
    double residual_nrm2 = 1.0;                         ///< Initialising residual value
    int iter             = 0;                           ///< Initialising iteration counter
    int max_iter         = 100000;                      ///< Setting maximum number of iterations
    
    std::cout << "Nx = " << Nx << ", Ny = " << Ny
          << ", Nz = " << Nz << ", epsilon = " << epsilon << std::endl;
    
    /// Stop if residual is smaller than tolerance or if maximum number of iterations is reached
    while (residual_nrm2 > epsilon && iter < max_iter) {
        jacobi(u, u_new, f, Nx, Ny, Nz, a, b, c, denom);
        F77NAME(dcopy)(N, u_new, 1, u, 1);
        residual_nrm2 = residual(u, f, Nx, Ny, Nz, a, b, c);
        ++iter;
    }
    
    /// Printing final residual
    std::cout << "Final residual = " << residual_nrm2 << std::endl;

    /// Free dynamic memory
    delete[] u;
    delete[] u_new;
    delete[] f;

    return 0;
}