/**
 * @file solver-mpi.cpp
 * @brief MPI implementation of the fused Jacobi solver.
 *
 * Provides the optimised jacobi_residual() routine which performs the
 * Jacobi update, residual computation and copy-back in a single pass
 * over the grid to reduce memory traffic.
 */

#include <cmath>
#include <mpi.h>
#include "solver.hpp"


/**
 * @brief Fused Jacobi sweep, residual computation and copy-back.
 *
 * Performs the Jacobi update, computes the residual on the current
 * solution, and copies the updated values back into @p u in a single
 * pass over the grid.  This reduces memory traffic by visiting each
 * grid point once instead of three separate passes.  The local
 * residual sums are combined across all ranks using MPI_Allreduce.
 *
 * @param u       Current solution array (local, including ghost layers).
 * @param u_new   Temporary array for the Jacobi update.
 * @param f       Forcing function array (local, including ghost layers).
 * @param Nx      Padded number of grid points in x-direction.
 * @param Ny      Padded number of grid points in y-direction.
 * @param Nz      Padded number of grid points in z-direction.
 * @param i_start First interior index in x (inclusive).
 * @param i_end   Last interior index in x (inclusive).
 * @param j_start First interior index in y (inclusive).
 * @param j_end   Last interior index in y (inclusive).
 * @param k_start First interior index in z (inclusive).
 * @param k_end   Last interior index in z (inclusive).
 * @param a       Finite-difference coefficient in x-direction.
 * @param b       Finite-difference coefficient in y-direction.
 * @param c       Finite-difference coefficient in z-direction.
 * @param denom   Precomputed denominator used in the Jacobi update.
 * @return double Global L2 norm of the residual.
 */
double jacobi_residual(double* u, double* u_new, const double* f,
                       int Nx, int Ny, int Nz,
                       int i_start, int i_end, int j_start, int j_end,
                       int k_start, int k_end,
                       double a, double b, double c, double denom) {

    double local_sum = 0.0;
    for (int k = k_start; k <= k_end; ++k) {
        for (int j = j_start; j <= j_end; ++j) {
            for (int i = i_start; i <= i_end; ++i) {

                int idx = i + Nx * (j + Ny * k);

                /// Jacobi update
                u_new[idx] = denom * (
                    (u[idx + 1]       + u[idx - 1])       * a +
                    (u[idx + Nx]      + u[idx - Nx])      * b +
                    (u[idx + Nx * Ny] + u[idx - Nx * Ny]) * c -
                    f[idx]);

                /// Residual on current solution
                double lap =
                    (u[idx + 1]       - 2.0 * u[idx] + u[idx - 1])       * a +
                    (u[idx + Nx]      - 2.0 * u[idx] + u[idx - Nx])      * b +
                    (u[idx + Nx * Ny] - 2.0 * u[idx] + u[idx - Nx * Ny]) * c;
                double r = f[idx] - lap;
                local_sum += r * r;

                /// Copy back
                u[idx] = u_new[idx];
            }
        }
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(global_sum);
}