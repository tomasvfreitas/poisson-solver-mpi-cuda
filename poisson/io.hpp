/**
 * @file io.hpp
 * @brief Declarations for file I/O routines used by the Poisson solver.
 *
 * Provides routines for reading a forcing function from a text file
 * and writing the converged solution to a text file.
 *
 * File format (both input and output):
 *   - First line: Nx Ny Nz (space-separated integers)
 *   - Subsequent Nx*Ny*Nz lines: x y z value (space-separated doubles)
 */

#ifndef IO_HPP
#define IO_HPP

#include <string>

/**
 * @brief Read a forcing function from a text file.
 *
 * Opens the specified file, reads Nx, Ny, Nz from the first line,
 * then reads Nx*Ny*Nz lines of (x, y, z, f) values and stores the
 * forcing values in the array @p f.
 *
 * The caller must pass uninitialised pointers for @p f; this routine
 * allocates the array using new[]. The caller is responsible for
 * freeing it with delete[].
 *
 * @param filename  Path to the input forcing file.
 * @param[out] f    Forcing function array (allocated by this routine).
 * @param[out] Nx   Number of grid points in x-direction.
 * @param[out] Ny   Number of grid points in y-direction.
 * @param[out] Nz   Number of grid points in z-direction.
 */
void read_forcing(const std::string& filename,
                  double*& f, int& Nx, int& Ny, int& Nz);

/**
 * @brief Write the solution to a text file.
 *
 * Writes Nx, Ny, Nz on the first line, followed by Nx*Ny*Nz lines
 * of (x, y, z, u) values.
 *
 * @param filename  Path to the output solution file.
 * @param u         Solution array.
 * @param Nx        Number of grid points in x-direction.
 * @param Ny        Number of grid points in y-direction.
 * @param Nz        Number of grid points in z-direction.
 */
void write_solution(const std::string& filename,
                    const double* u, int Nx, int Ny, int Nz);

#endif
