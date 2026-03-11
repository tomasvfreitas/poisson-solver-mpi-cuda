#include <iostream>

#define F77NAME(x) x##_
extern "C"{
    void F77NAME(dcopy)(const int& N,
                        const double* x, const int& incx,
                        double* y, const int& incy); 
                         
    void F77NAME(daxpy)(const int& N, const double& alpha,
                          const double* x, const int& incx,
                          double* y, const int& incy);
                           
    double F77NAME(dasum)(const int& N, const double* x, const int& incx);
}

int main() {
    
    int Nx = 4;
    int Ny = 4;
    int Nz = 4;
    
    double *u     = new double[Nx * Ny * Nz]();
    double *u_new = new double[Nx * Ny * Nz]();
    double *f     = new double[Nx * Ny * Nz]();
    
    double hx = 1.0 / (Nx - 1.0);
    double hy = 1.0 / (Ny - 1.0);
    double hz = 1.0 / (Nz - 1.0);
    
    double a = 1.0 / (hx * hx);
    double b = 1.0 / (hy * hy);
    double c = 1.0 / (hz * hz);
    
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
    
    
    //double *residual = new double[Nx * Ny * Nz]();
    //double residual_nrm2 = 0.0;
    //F77NAME(dcopy)(Nx * Ny * Nz, f, 1, residual, 1);
    //for (int idx = 0; idx < Nx * Ny * Nz; ++idx) {
    //    F77NAME(daxpy)(Nx * Ny * Nz, -1.0, u_new, 1, residual, 1);
    //}
    //residual_nrm2 = F77NAME(dasum)();
    //residual_nrm2 = sqrt(residual_nrm2);
    
    
    //double epsilon = 1e-8;
    double denom = 1 / (2 * (a + b + c));
    //while (residual_nrm2 > epsilon) {
    for (int w = 0; w < 10; ++w) {
        for (int k = 1; k < Nz - 1; ++k) {
            for (int j = 1; j < Ny - 1; ++j) {
                for (int i = 1; i < Nx - 1; ++i) {
                    int idx = i + Nx * (j + Ny * k);
                    u_new[idx] = denom * ((u[idx + 1] + u[idx - 1]) * a + (u[idx + Nx] + u[idx - Nx]) * b + (u[idx + Nx*Ny] + u[idx - Nx*Ny]) * c - f[idx]);
                    //residual_rnm2 = f - u;
                }
            }
        }
    F77NAME(dcopy)(Nx*Ny*Nz, u_new, 1, u, 1);
    }


    int idx111 = 1 + Nx * (1 + Ny * 1);
    int idx222 = 2 + Nx * (2 + Ny * 2);

    std::cout << "u(1,1,1) old = " << u[idx111] << std::endl;
    std::cout << "u(1,1,1) new = " << u_new[idx111] << std::endl;
 
    std::cout << "u(2,2,2) old = " << u[idx222] << std::endl;
    std::cout << "u(2,2,2) new = " << u_new[idx222] << std::endl;


    //for (int idx = 0; idx < Nx * Ny * Nz; ++idx) {
    //    u[idx] = u_new[idx];
    //}
    
    // Free dynamic memory
    delete[] u;
    delete[] u_new;
    delete[] f;
    
	return 0;
}
