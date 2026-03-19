#ifndef SOLVER_HPP
#define SOLVER_HPP

/**
 * @file solver.hpp
 * @brief Declarations for the Jacobi solver and residual computation.
 */


/**
 * @brief Perform one Jacobi sweep over the interior grid points (serial version).
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
            double a, double b, double c, double denom);

/**
 * @brief Compute the L2 norm of the residual (serial version).
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
                double a, double b, double c);

/**
 * @brief Fused Jacobi sweep, residual computation and copy-back (MPI version).
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
                       double a, double b, double c, double denom);
                       
                       
#ifdef CUDA_ENABLED
/**
 * @brief CUDA version of the fused Jacobi sweep, residual and copy-back.
 *
 * Offloads the computation to the GPU. Each grid point is processed
 * by a separate CUDA thread. The local residual sum is combined
 * across MPI ranks using MPI_Allreduce.
 *
 * @param d_u     Current solution array (unified managed memory).
 * @param d_u_new Temporary array for the Jacobi update.
 * @param d_f     Forcing function array (unified managed memory).
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
double jacobi_residual_cuda(double* d_u, double* d_u_new, const double* d_f,
                            int Nx, int Ny, int Nz,
                            int i_start, int i_end,
                            int j_start, int j_end,
                            int k_start, int k_end,
                            double a, double b, double c, double denom);
#endif

#endif