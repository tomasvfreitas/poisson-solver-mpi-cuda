#include <cmath>
#include <mpi.h>
#include "solver.hpp"

/**
 * @brief Perform one Jacobi sweep over the local interior grid points.
 *
 * Updates the interior values of the local solution using
 * finite-differences for the 3D Poisson equation.
 * Boundary and ghost layer values are not modified.
 *
 * @param u Current solution array (local, including ghost layers).
 * @param u_new Updated solution array (local, including ghost layers).
 * @param f Forcing function array (local, including ghost layers).
 * @param Nx Local number of grid points in x-direction.
 * @param Ny Local number of grid points in y-direction.
 * @param Nz Local number of grid points in z-direction.
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
 * @brief Compute the global L2 norm of the residual.
 *
 * Each rank evaluates the discrete residual
 * \f$ r = f - \nabla_h^2 u \f$
 * over its local interior grid points. The local sums are
 * combined across all ranks using MPI_Allreduce, and the
 * global Euclidean norm is returned.
 *
 * @param u Current solution array (local, including ghost layers).
 * @param f Forcing function array (local, including ghost layers).
 * @param Nx Local number of grid points in x-direction.
 * @param Ny Local number of grid points in y-direction.
 * @param Nz Local number of grid points in z-direction.
 * @param a Finite-difference coefficient in x-direction.
 * @param b Finite-difference coefficient in y-direction.
 * @param c Finite-difference coefficient in z-direction.
 * @return double Global L2 norm of the residual.
 */
double residual(const double* u, const double* f,
                int Nx, int Ny, int Nz,
                double a, double b, double c) {
                    
    double local_sum = 0.0;

    for (int k = 1; k < Nz - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {

                int idx = i + Nx * (j + Ny * k);

                double lap =
                    (u[idx + 1]       - 2.0 * u[idx] + u[idx - 1])       * a +
                    (u[idx + Nx]      - 2.0 * u[idx] + u[idx - Nx])      * b +
                    (u[idx + Nx * Ny] - 2.0 * u[idx] + u[idx - Nx * Ny]) * c;

                double r = f[idx] - lap;
                local_sum += r * r;
            }
        }
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(global_sum);
}