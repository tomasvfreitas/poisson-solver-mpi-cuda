/**
 * @file io.cpp
 * @brief File I/O routines for the Poisson solver.
 *
 * Implements reading of a forcing function from a text file and
 * writing of the converged solution to a text file.
 *
 * File format (both input and output):
 *   - First line: Nx Ny Nz (space-separated integers)
 *   - Subsequent Nx*Ny*Nz lines: x y z (f or u) (space-separated doubles)
 */

#include "io.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>


/**
 * @brief Read a forcing function from a text file.
 *
 * Opens the specified file, reads Nx, Ny, Nz from the first line,
 * then reads Nx*Ny*Nz lines of (x, y, z, f) values. The forcing
 * values are stored in row-major order consistent with the index
 * mapping idx = i + Nx * (j + Ny * k).
 *
 * The grid coordinates in the file are used to determine the
 * (i, j, k) indices via the grid spacings hx, hy, hz.
 *
 * @param filename  Path to the input forcing file.
 * @param[out] f    Forcing function array (allocated by this routine).
 * @param[out] Nx   Number of grid points in x-direction.
 * @param[out] Ny   Number of grid points in y-direction.
 * @param[out] Nz   Number of grid points in z-direction.
 *
 * @throws std::runtime_error if the file cannot be opened or read.
 */
void read_forcing(const std::string& filename,
                  double*& f, int& Nx, int& Ny, int& Nz) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open forcing file: " + filename);
    }

    infile >> Nx >> Ny >> Nz;
    if (infile.fail()) {
        throw std::runtime_error("Failed to read Nx, Ny, Nz from: " + filename);
    }

    int N = Nx * Ny * Nz;
    f = new double[N]();

    double hx = 1.0 / (Nx - 1.0);
    double hy = 1.0 / (Ny - 1.0);
    double hz = 1.0 / (Nz - 1.0);

    for (int n = 0; n < N; ++n) {
        double x, y, z, val;
        infile >> x >> y >> z >> val;

        if (infile.fail()) {
            delete[] f;
            f = nullptr;
            throw std::runtime_error("Failed to read line " +
                                     std::to_string(n + 2) +
                                     " from: " + filename);
        }

        /// Convert coordinates to grid indices
        int i = static_cast<int>(std::round(x / hx));
        int j = static_cast<int>(std::round(y / hy));
        int k = static_cast<int>(std::round(z / hz));

        int idx = i + Nx * (j + Ny * k);
        f[idx] = val;
    }

    infile.close();
}


/**
 * @brief Write the solution to a text file.
 *
 * Writes Nx, Ny, Nz on the first line, followed by Nx*Ny*Nz lines
 * of (x, y, z, u) values in the same ordering used by the solver
 * (k outer, j middle, i inner).
 *
 * @param filename  Path to the output solution file.
 * @param u         Solution array.
 * @param Nx        Number of grid points in x-direction.
 * @param Ny        Number of grid points in y-direction.
 * @param Nz        Number of grid points in z-direction.
 *
 * @throws std::runtime_error if the file cannot be opened.
 */
void write_solution(const std::string& filename,
                    const double* u, int Nx, int Ny, int Nz) {

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }

    outfile << Nx << " " << Ny << " " << Nz << std::endl;

    double hx = 1.0 / (Nx - 1.0);
    double hy = 1.0 / (Ny - 1.0);
    double hz = 1.0 / (Nz - 1.0);

    outfile << std::setprecision(10);

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = i + Nx * (j + Ny * k);

                double x = i * hx;
                double y = j * hy;
                double z = k * hz;

                outfile << x << " " << y << " " << z << " "
                        << u[idx] << std::endl;
            }
        }
    }

    outfile.close();
}
