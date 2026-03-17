/**
 * @file halo-mpi.hpp
 * @brief Declaration of the halo exchange routine for the MPI Poisson solver.
 */

#ifndef HALO_HPP
#define HALO_HPP

/**
 * @brief Perform halo exchange between neighbouring MPI ranks.
 *
 * Exchanges ghost layer data in all three directions using
 * MPI_Sendrecv. For the x-direction, data is strided. For
 * y-direction, data is also strided. For z-direction, data
 * is contiguous (a full xy-plane).
 *
 * @param u         Local solution array (including ghost layers).
 * @param loc_Nx    Local grid points in x (excluding ghosts).
 * @param loc_Ny    Local grid points in y (excluding ghosts).
 * @param loc_Nz    Local grid points in z (excluding ghosts).
 * @param nbr_xm    Rank of neighbour in -x direction (-1 if none).
 * @param nbr_xp    Rank of neighbour in +x direction (-1 if none).
 * @param nbr_ym    Rank of neighbour in -y direction (-1 if none).
 * @param nbr_yp    Rank of neighbour in +y direction (-1 if none).
 * @param nbr_zm    Rank of neighbour in -z direction (-1 if none).
 * @param nbr_zp    Rank of neighbour in +z direction (-1 if none).
 */
void halo_exchange(double* u,
                   int loc_Nx, int loc_Ny, int loc_Nz,
                   int nbr_xm, int nbr_xp,
                   int nbr_ym, int nbr_yp,
                   int nbr_zm, int nbr_zp);

#endif