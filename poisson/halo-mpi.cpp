/**
 * @file halo-mpi.cpp
 * @brief Halo exchange routine for the MPI Poisson solver.
 *
 * Implements the ghost layer exchange between neighbouring MPI
 * ranks in all three spatial directions. Used by both the main
 * solver and the MPI test suite.
 */
#include "halo-mpi.hpp"
#include <mpi.h>

void halo_exchange(double* u,
                   int loc_Nx, int loc_Ny, int loc_Nz,
                   int nbr_xm, int nbr_xp,
                   int nbr_ym, int nbr_yp,
                   int nbr_zm, int nbr_zp) {

    /// Padded dimensions (including ghost layers)
    int pNx = loc_Nx + 2;
    int pNy = loc_Ny + 2;
    int pNz = loc_Nz + 2;

    /// Indexing macro: idx(i,j,k) in padded array
    #define IDX(i,j,k) ((i) + pNx * ((j) + pNy * (k)))

    // x-direction exchange
    {
        /// One column in j: loc_Ny elements, stride pNx
        MPI_Datatype x_col;
        MPI_Type_vector(loc_Ny, 1, pNx, MPI_DOUBLE, &x_col);
        MPI_Type_commit(&x_col);

        /// Full x-face: loc_Nz such columns, stride pNx*pNy bytes apart
        MPI_Datatype x_face;
        MPI_Type_create_hvector(loc_Nz, 1, pNx * pNy * sizeof(double), x_col, &x_face);
        MPI_Type_commit(&x_face);

        /// Send left face (i=1) to nbr_xm, receive into right ghost (i=pNx-1) from nbr_xp
        MPI_Sendrecv(&u[IDX(1, 1, 1)],      1, x_face, nbr_xm >= 0 ? nbr_xm : MPI_PROC_NULL, 0,
                     &u[IDX(pNx-1, 1, 1)],   1, x_face, nbr_xp >= 0 ? nbr_xp : MPI_PROC_NULL, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /// Send right face (i=loc_Nx) to nbr_xp, receive into left ghost (i=0) from nbr_xm
        MPI_Sendrecv(&u[IDX(loc_Nx, 1, 1)],  1, x_face, nbr_xp >= 0 ? nbr_xp : MPI_PROC_NULL, 1,
                     &u[IDX(0, 1, 1)],        1, x_face, nbr_xm >= 0 ? nbr_xm : MPI_PROC_NULL, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Type_free(&x_face);
        MPI_Type_free(&x_col);
    }

    // y-direction exchange
    {
        /// y-face: loc_Nz blocks of loc_Nx contiguous elements, stride pNx*pNy
        MPI_Datatype y_face;
        MPI_Type_vector(loc_Nz, loc_Nx, pNx * pNy, MPI_DOUBLE, &y_face);
        MPI_Type_commit(&y_face);

        /// Send bottom face (j=1) to nbr_ym, receive into top ghost (j=pNy-1) from nbr_yp
        MPI_Sendrecv(&u[IDX(1, 1, 1)],        1, y_face, nbr_ym >= 0 ? nbr_ym : MPI_PROC_NULL, 2,
                     &u[IDX(1, pNy-1, 1)],     1, y_face, nbr_yp >= 0 ? nbr_yp : MPI_PROC_NULL, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /// Send top face (j=loc_Ny) to nbr_yp, receive into bottom ghost (j=0) from nbr_ym
        MPI_Sendrecv(&u[IDX(1, loc_Ny, 1)],   1, y_face, nbr_yp >= 0 ? nbr_yp : MPI_PROC_NULL, 3,
                     &u[IDX(1, 0, 1)],         1, y_face, nbr_ym >= 0 ? nbr_ym : MPI_PROC_NULL, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Type_free(&y_face);
    }

    // z-direction exchange
    {
        /// z-face: loc_Ny rows of loc_Nx contiguous elements, stride pNx
        MPI_Datatype z_face;
        MPI_Type_vector(loc_Ny, loc_Nx, pNx, MPI_DOUBLE, &z_face);
        MPI_Type_commit(&z_face);

        /// Send front face (k=1) to nbr_zm, receive into back ghost (k=pNz-1) from nbr_zp
        MPI_Sendrecv(&u[IDX(1, 1, 1)],        1, z_face, nbr_zm >= 0 ? nbr_zm : MPI_PROC_NULL, 4,
                     &u[IDX(1, 1, pNz-1)],     1, z_face, nbr_zp >= 0 ? nbr_zp : MPI_PROC_NULL, 4,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /// Send back face (k=loc_Nz) to nbr_zp, receive into front ghost (k=0) from nbr_zm
        MPI_Sendrecv(&u[IDX(1, 1, loc_Nz)],   1, z_face, nbr_zp >= 0 ? nbr_zp : MPI_PROC_NULL, 5,
                     &u[IDX(1, 1, 0)],         1, z_face, nbr_zm >= 0 ? nbr_zm : MPI_PROC_NULL, 5,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Type_free(&z_face);
}

    #undef IDX
}