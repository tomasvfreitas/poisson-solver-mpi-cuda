#ifndef SOLVER_HPP
#define SOLVER_HPP

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
            double a, double b, double c, double denom);

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
                double a, double b, double c);

#endif