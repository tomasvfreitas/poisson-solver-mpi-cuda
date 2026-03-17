#ifndef TEST_CASE_HPP
#define TEST_CASE_HPP
/**
 * @file test-case.hpp
 * @brief Declarations for built-in Poisson solver test case initialisation.
 *
 * This header declares the routines used to initialise the numerical
 * solution array, temporary solution array, and forcing array for the
 * built-in 3D Poisson test cases, in both serial and parallel contexts.
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


/**
 * @brief Initialise a local subdomain for one of the built-in test cases.
 *
 * This routine sets the local portion of the numerical solution array
 * @p u, the temporary solution array @p u_new, and the forcing array
 * @p f for use in parallel (MPI) execution.
 *
 * Coordinates are computed from the global starting indices so that
 * boundary conditions and forcing values correspond to the correct
 * position in the global domain. Boundary detection uses the global
 * grid dimensions.
 *
 * @param test_case Integer identifier of the test case to initialise.
 * @param u Local numerical solution array.
 * @param u_new Local temporary solution array used for Jacobi updates.
 * @param f Local forcing function array.
 * @param loc_Nx Local number of grid points in the x-direction.
 * @param loc_Ny Local number of grid points in the y-direction.
 * @param loc_Nz Local number of grid points in the z-direction.
 * @param start_x Global starting index of this subdomain in x.
 * @param start_y Global starting index of this subdomain in y.
 * @param start_z Global starting index of this subdomain in z.
 * @param Nx Global number of grid points in the x-direction.
 * @param Ny Global number of grid points in the y-direction.
 * @param Nz Global number of grid points in the z-direction.
 * @param hx Grid spacing in the x-direction.
 * @param hy Grid spacing in the y-direction.
 * @param hz Grid spacing in the z-direction.
 */
void initialise_test_case_local(int test_case,
                                double* u, double* u_new, double* f,
                                int loc_Nx, int loc_Ny, int loc_Nz,
                                int start_x, int start_y, int start_z,
                                int Nx, int Ny, int Nz,
                                double hx, double hy, double hz);
#endif