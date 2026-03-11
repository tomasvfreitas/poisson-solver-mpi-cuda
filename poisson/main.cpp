/**
 * @file main.cpp
 * @brief Serial Jacobi solver for the 3D Poisson equation.
 *
 * This program solves the Poisson equation
 *
 *      \f$ \nabla^2 u = f \f$
 *
 * on a 3-dimensional grid using Jacobi iteration with Dirichlet boundary
 * conditions. The solver iterates until the L2 norm of the residual falls
 * below a specified tolerance.
 *
 * The solver computes the numerical solution and monitors convergence
 * using the residual norm.
 */

#include <iostream>
#include <cmath>

#define F77NAME(x) x##_
extern "C"{
    void F77NAME(dcopy)(const int& N,
                        const double* x, const int& incx,
                        double* y, const int& incy); 
                         
    //void F77NAME(daxpy)(const int& N, const double& alpha, const double* x, const int& incx, double* y, const int& incy);
                           
    //double F77NAME(dasum)(const int& N, const double* x, const int& incx);
}


/**
 * @brief Entry point for the serial Poisson solver.
 *
 * The program:
 * 1. Initialises the computational grid.
 * 2. Applies Dirichlet boundary conditions.
 * 3. Performs Jacobi iterations.
 * 4. Computes the residual norm after each iteration.
 * 5. Stops when the residual falls below the specified tolerance.
 *
 * @return int Program exit status.
 */
 
 int main() {
    
    /// Number of grid points in x-direction
    int Nx = 4;
    
    /// Number of grid points in y-direction
    int Ny = 4;
    
    /// Number of grid points in z-direction
    int Nz = 4;
    
    
    /// Numerical solution
    double *u     = new double[Nx * Ny * Nz]();
    
    /// Temporary array to update solution
    double *u_new = new double[Nx * Ny * Nz]();
    
    /// Forcing function
    double *f     = new double[Nx * Ny * Nz]();
    
    
    /// Grid spacing in x-direction
    double hx = 1.0 / (Nx - 1.0);
    
    /// Grid spacing in y-direction
    double hy = 1.0 / (Ny - 1.0);
    
    /// Grid spacing in z-direction
    double hz = 1.0 / (Nz - 1.0);
    
    
    /// Finite-difference coefficient in x-direction
    double a = 1.0 / (hx * hx);

    /// Finite-difference coefficient in y-direction
    double b = 1.0 / (hy * hy);

    /// Finite-difference coefficient in z-direction
    double c = 1.0 / (hz * hz);
    
    
    /// Enforcing Dirichlet boundary conditions and initialising interior values
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                
                int idx = i + Nx * (j + Ny * k);
                
                double x = i * hx;
                double y = j * hy;
                double z = k * hz;
                
                if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1) {
                    u[idx] = x*x + y*y + z*z;
                }
                else {
                    u[idx] = 0.0;
                }
                
                f[idx]     = 6.0;
                u_new[idx] = u[idx];
            }
        }
    }

    
    double epsilon       = 1e-8;
    double denom         = 1 / (2 * (a + b + c));
    double residual_nrm2 = 1.0;
    while (residual_nrm2 > epsilon) {
        
        /// Performing the Jacobi iteration on the interior grid points
        for (int k = 1; k < Nz - 1; ++k) {
            for (int j = 1; j < Ny - 1; ++j) {
                for (int i = 1; i < Nx - 1; ++i) {
                    int idx = i + Nx * (j + Ny * k);
                    u_new[idx] = denom * ((u[idx + 1] + u[idx - 1]) * a +
                                 (u[idx + Nx] + u[idx - Nx]) * b +
                                 (u[idx + Nx*Ny] + u[idx - Nx*Ny]) * c - f[idx]);
                }
            }
        }
        
        /// Updating the numerical solution
        F77NAME(dcopy)(Nx*Ny*Nz, u_new, 1, u, 1);
        
        /// Residual calculation
        double sum      = 0.0;
        double residual = 0.0;
        for (int k = 1; k < Nz - 1; ++k) {
            for (int j = 1; j < Ny - 1; ++j) {
                for (int i = 1; i < Nx - 1; ++i) {
                    int idx = i + Nx * (j + Ny * k);
                    double lap = (u[idx + 1] - 2 * u[idx] + u[idx - 1]) * a +
                                 (u[idx + Nx] - 2 * u[idx] + u[idx - Nx]) * b +
                                 (u[idx + Nx*Ny] - 2 * u[idx] + u[idx - Nx*Ny]) * c;
                    residual = f[idx] - lap;
                    sum += residual * residual;
                    
                }
            }
        }
        residual_nrm2 = sqrt(sum);
    }
    
    /// Printing out final residual to terminal
    std::cout << "Final residual = " << residual_nrm2 << std::endl;

    /// Free dynamic memory
    delete[] u;
    delete[] u_new;
    delete[] f;
    
	return 0;
}
