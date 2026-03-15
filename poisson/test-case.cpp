#include "test-case.hpp"

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
void test_case1(double* u, double* u_new, double* f,
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


void test_case2(double* u, double* u_new, double* f,
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
                    u[idx] = sin(M_PI*x) * sin(M_PI*y) * sin(M_PI*z);
                } else {
                    u[idx] = 0.0;
                }

                f[idx]     = -3 * M_PI * M_PI * sin(M_PI*x) * sin(M_PI*y) * sin(M_PI*z);
                u_new[idx] = u[idx];
            }
        }
    }
}


void test_case3(double* u, double* u_new, double* f,
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
                    u[idx] = sin(M_PI*x) * sin(4*M_PI*y) * sin(8*M_PI*z);
                } else {
                    u[idx] = 0.0;
                }

                f[idx]     = -81 * M_PI * M_PI * sin(M_PI*x) * sin(4*M_PI*y) * sin(8*M_PI*z);
                u_new[idx] = u[idx];
            }
        }
    }
}


void test_case4(double* u, double* u_new, double* f,
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

void test_case5(double* u, double* u_new, double* f,
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