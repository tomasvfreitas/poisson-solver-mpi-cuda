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
 * @file main.cpp
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

namespace po = boost::program_options;

#define F77NAME(x) x##_
extern "C" {
    void F77NAME(dcopy)(const int& N,
                        const double* x, const int& incx,
                        double* y, const int& incy);
}


/**
 * @brief Initialise the verification problem on the 3D grid.
 *
 * Applies Dirichlet boundary conditions using the exact solution
 * \f$ u(x,y,z) = x^2 + y^2 + z^2 \f$, sets the forcing term to \f$ f = 6 \f$,
 * and initialises interior values to zero.
 *
 * @param u Numerical solution array.
 * @param u_new Temporary solution array used for Jacobi updates.
 * @param f Forcing function array.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 * @param hx Grid spacing in x-direction.
 * @param hy Grid spacing in y-direction.
 * @param hz Grid spacing in z-direction.
 */
void initialise_problem(double* u, double* u_new, double* f,
                        int Nx, int Ny, int Nz, double hx,
                        double hy, double hz) {
 
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {

                int idx = i + Nx * (j + Ny * k);

                double x = i * hx;
                double y = j * hy;
                double z = k * hz;

                if (i == 0 || i == Nx - 1 ||
                    j == 0 || j == Ny - 1 ||
                    k == 0 || k == Nz - 1) {
                    u[idx] = x * x + y * y + z * z;
                } else {
                    u[idx] = 0.0;
                }

                f[idx]     = 6.0;
                u_new[idx] = u[idx];
            }
        }
    }
}


/**
 * @brief Perform one Jacobi sweep over the interior grid points.
 *
 * Updates the interior values of the solution using
 * finite-differences for the 3D Poisson equation.
 * Boundary values are not modified.
 *
 * @param u Current solution array.
 * @param u_new Updated solution array.
 * @param f Forcing function array.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 * @param a Finite-difference coefficient in x-direction.
 * @param b Finite-difference coefficient in y-direction.
 * @param c Finite-difference coefficient in z-direction.
 * @param denom Precomputed denominator used in the Jacobi update.
 */
void jacobi(double* u, double* u_new, const double* f,
            int Nx, int Ny, int Nz,
            double a, double b, double c, double denom) {
                
    for (int k = 1; k < Nz - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {

                int idx = i + Nx * (j + Ny * k);

                u_new[idx] = denom * (
                    (u[idx + 1]       + u[idx - 1])       * a +
                    (u[idx + Nx]      + u[idx - Nx])      * b +
                    (u[idx + Nx * Ny] + u[idx - Nx * Ny]) * c -
                    f[idx]);
            }
        }
    }
}


/**
 * @brief Compute the L2 norm of the residual.
 *
 * Evaluates the discrete residual
 * \f$ r = f - \nabla_h^2 u \f$
 * over the interior grid points and returns its Euclidean norm.
 *
 * @param u Current solution array.
 * @param f Forcing function array.
 * @param Nx Number of grid points in x-direction.
 * @param Ny Number of grid points in y-direction.
 * @param Nz Number of grid points in z-direction.
 * @param a Finite-difference coefficient in x-direction.
 * @param b Finite-difference coefficient in y-direction.
 * @param c Finite-difference coefficient in z-direction.
 * @return double L2 norm of the residual.
 */
double residual(const double* u, const double* f,
                int Nx, int Ny, int Nz,
                double a, double b, double c) {
                    
    double sum = 0.0;

    for (int k = 1; k < Nz - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {

                int idx = i + Nx * (j + Ny * k);

                double lap =
                    (u[idx + 1]       - 2.0 * u[idx] + u[idx - 1])       * a +
                    (u[idx + Nx]      - 2.0 * u[idx] + u[idx - Nx])      * b +
                    (u[idx + Nx * Ny] - 2.0 * u[idx] + u[idx - Nx * Ny]) * c;

                double r = f[idx] - lap;
                sum += r * r;
            }
        }
    }

    return std::sqrt(sum);
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
    
    /// Check if user input a forcing file and assign value
    bool has_test    = vm.count("test") && !vm["test"].defaulted();
    bool has_forcing = vm.count("forcing");
    int  test_case   = vm["test"].as<int>();
    std::string forcing_file;
    
   // if (has_forcing) {
   //     forcing_file = vm["forcing"].as<std::string>();
   //     std::cerr << "Error: --forcing mode not implemented yet." << std::endl;
   //     return 1;
   // }
    
    /// Check if user provided both forcing function and selected a test case and give error
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
    
    /// Enforcing Dirichlet boundary conditions and initialising interior values
    /// Temporary since only case 1 implemented yet
    if (has_test) {
        if (test_case != 1) {
            std::cerr << "Error: only test case 1 implemented so far." << std::endl;
            return 1;
        }
        initialise_problem(u, u_new, f, Nx, Ny, Nz, hx, hy, hz);
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