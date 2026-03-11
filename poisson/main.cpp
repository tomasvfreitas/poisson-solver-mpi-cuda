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
int main() {
    
    int Nx = 4;                         ///< Number of grid points in x-direction
    int Ny = 4;                         ///< Number of grid points in y-direction
    int Nz = 4;                         ///< Number of grid points in z-direction
    
    int N  = Nx * Ny * Nz;              ///< Nx by Ny by Nz dimensions for simplicity

    double* u     = new double[N]();    ///< Numerical solution
    double* u_new = new double[N]();    ///< Temporary array to update solution
    double* f     = new double[N]();    ///< Forcing function

    double hx = 1.0 / (Nx - 1.0);       ///< Grid spacing in x-direction
    double hy = 1.0 / (Ny - 1.0);       ///< Grid spacing in y-direction
    double hz = 1.0 / (Nz - 1.0);       ///< Grid spacing in z-direction

    double a = 1.0 / (hx * hx);         ///< Finite-difference coefficient in x-direction
    double b = 1.0 / (hy * hy);         ///< Finite-difference coefficient in y-direction
    double c = 1.0 / (hz * hz);         ///< Finite-difference coefficient in z-direction

    /// Enforcing Dirichlet boundary conditions and initialising interior values
    initialise_problem(u, u_new, f, Nx, Ny, Nz, hx, hy, hz);
    
    double epsilon       = 1e-8;        ///< Tolerance
    double denom         = 1.0 / (2.0 * (a + b + c));   ///< Setting denom for simplicity
    double residual_nrm2 = 1.0;         ///< Initialising residual value
    int iter             = 0;           ///< Initialising iteration counter
    int max_iter         = 100000;      ///< Setting maximum number of iterations
    
    
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