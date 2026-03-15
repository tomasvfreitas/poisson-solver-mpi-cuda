#ifndef TEST_CASE_HPP
#define TEST_CASE_HPP


/**
 * @file test-case.hpp
 * @brief Declarations for built-in Poisson solver test case initialisation.
 *
 * This header declares the routine used to initialise the numerical
 * solution array, temporary solution array, and forcing array for the
 * built-in 3D Poisson test cases.
 */


/**
 * @brief Initialise one of the built-in Poisson test cases.
 *
 * This routine sets the numerical solution array @p u, the temporary
 * solution array @p u_new, and the forcing array @p f according to the
 * selected built-in test case.
 *
 * Boundary values are applied on the outer faces of the 3D domain,
 * interior values are initialised to zero, and @p u_new is initialised
 * to match @p u.
 *
 * @param test_case Integer identifier of the test case to initialise.
 * @param u Numerical solution array.
 * @param u_new Temporary solution array used for Jacobi updates.
 * @param f Forcing function array.
 * @param Nx Number of grid points in the x-direction.
 * @param Ny Number of grid points in the y-direction.
 * @param Nz Number of grid points in the z-direction.
 * @param hx Grid spacing in the x-direction.
 * @param hy Grid spacing in the y-direction.
 * @param hz Grid spacing in the z-direction.
 */
void initialise_test_case(int test_case,
                          double* u, double* u_new, double* f,
                          int Nx, int Ny, int Nz,
                          double hx, double hy, double hz);

#endif