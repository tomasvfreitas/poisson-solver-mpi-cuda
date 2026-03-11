#include <iostream>

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
    
    //double x = i * hx;
    //double y = j * hy;
    //double z = k * hz;
    
    for (int k = 1; k < Nz - 1; ++k) {
        for (int j = 1; j < Ny - 1; ++j) {
            for (int i = 1; i < Nx - 1; ++i) {
                int idx = i + Nx * (j + Ny * k);
                f[idx] = 0.0;
                u_new[idx] = (1 / (2*(a + b + c))) * ((u[idx + 1] + u[idx - 1]) * a + (u[idx + Nx] + u[idx - Nx]) * b + (u[idx + Nx*Ny] + u[idx - Nx*Ny]) * c - f[idx]);
            }
        }
    }

    // Free dynamic memory
    delete[] u;
    delete[] u_new;
    delete[] f;
    
	return 0;
}
