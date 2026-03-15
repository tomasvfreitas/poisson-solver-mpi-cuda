#ifndef TEST_CASE_HPP
#define TEST_CASE_HPP

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
                        double hy, double hz);

#endif