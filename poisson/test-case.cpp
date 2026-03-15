/**
 * @file test-case.cpp
 * @brief Initialisation routines for built-in Poisson solver test cases.
 *
 * This file defines the built-in test cases used to initialise the
 * numerical solution array, temporary solution array, and forcing
 * function array for the 3D Poisson solver.
 *
 * A single routine, initialise_test_case(), is provided to select
 * the requested test case and populate the arrays accordingly.
 */
 
 
#include "test-case.hpp"
#include <cmath>
#include <stdexcept>


/**
 * @brief Check whether a grid point lies on the boundary of the domain.
 *
 * A point is considered to lie on the boundary if its index is on any
 * of the six faces of the 3D grid.
 *
 * @param i Grid index in the x-direction.
 * @param j Grid index in the y-direction.
 * @param k Grid index in the z-direction.
 * @param Nx Number of grid points in the x-direction.
 * @param Ny Number of grid points in the y-direction.
 * @param Nz Number of grid points in the z-direction.
 * @return true if the point lies on the boundary, false otherwise.
 */
namespace {
    bool is_boundary(int i, int j, int k, int Nx, int Ny, int Nz) {
        return (i == 0 || i == Nx - 1 ||
                j == 0 || j == Ny - 1 ||
                k == 0 || k == Nz - 1);
    }
}


/**
 * @brief Initialise one of the built-in Poisson test cases.
 *
 * This routine sets the initial solution array @p u, the temporary
 * solution array @p u_new, and the forcing array @p f according to
 * the selected test case.
 *
 * The following cases are supported:
 * - Case 1: \f$ u(x,y,z) = x^2 + y^2 + z^2 \f$, with \f$ f = 6 \f$
 * - Case 2: \f$ u(x,y,z) = \sin(\pi x)\sin(\pi y)\sin(\pi z) \f$
 * - Case 3: \f$ u(x,y,z) = \sin(\pi x)\sin(4\pi y)\sin(8\pi z) \f$
 * - Case 4: Zero Dirichlet boundary conditions with Gaussian forcing
 * - Case 5: Zero Dirichlet boundary conditions with piecewise forcing
 *
 * Boundary values are set from the selected boundary function, while
 * interior values are initialised to zero. The temporary array @p u_new
 * is initialised to match @p u.
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
 *
 * @throws std::invalid_argument if @p test_case is not in the range 1-5.
 */
void initialise_test_case(int test_case,
                          double* u, double* u_new, double* f,
                          int Nx, int Ny, int Nz,
                          double hx, double hy, double hz) {
    
    double (*boundary)(double, double, double) = nullptr;
    double (*forcing)(double, double, double) = nullptr;
    
    switch(test_case) {
        case 1:
            boundary = [](double x, double y, double z) {
               return x * x + y * y + z * z;
            };
            forcing = [](double, double, double) {
                return 6.0;
            };
            break;

        case 2:
            boundary = [](double x, double y, double z) {
                return std::sin(M_PI * x) * std::sin(M_PI * y) 
                                          * std::sin(M_PI * z);
            };
            forcing = [](double x, double y, double z) {
                return -3.0 * M_PI * M_PI * std::sin(M_PI * x) 
                                          * std::sin(M_PI * y) 
                                          * std::sin(M_PI * z);
            };
            break;

        case 3:
            boundary = [](double x, double y, double z) {
                return std::sin(M_PI * x) * std::sin(4.0 * M_PI * y) 
                                          * std::sin(8.0 * M_PI * z);
            };
            forcing = [](double x, double y, double z) {
                return -81.0 * M_PI * M_PI * std::sin(M_PI * x) 
                                           * std::sin(4.0 * M_PI * y) 
                                           * std::sin(8.0 * M_PI * z);
            };
            break;

        case 4:
            boundary = [](double, double, double) {
                return 0.0;
            };
            forcing = [](double x, double y, double z) {
                return 100.0 * std::exp(-100.0*((x - 0.5) * (x - 0.5) +
                                                (y - 0.5) * (y - 0.5) +
                                                (z - 0.5) * (z - 0.5)));
            };
            break;

        case 5:
            boundary = [](double, double, double) {
                return 0.0;
            };
            forcing = [](double x, double, double) {
                return (x < 0.5) ? 1.0 : -1.0;
            };
            break;
        
        default:
            throw std::invalid_argument("Invalid test case");
    }

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                
                int idx = i + Nx * (j + Ny * k);

                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                
                if (is_boundary(i, j, k, Nx, Ny, Nz)) {
                    u[idx] = boundary(x, y, z);
                }
                else {
                    u[idx] = 0.0;
                }
                f[idx] = forcing(x, y, z);
                u_new[idx] = u[idx];
            }
        }
    }
}  