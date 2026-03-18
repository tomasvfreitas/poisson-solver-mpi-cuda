/**
 * @file test-mpi.cpp
 * @brief MPI test suite for the parallel Jacobi Poisson solver.
 *
 * Executes the three verification cases (1-3) in parallel using the
 * same domain decomposition as main-mpi.cpp and checks that:
 *   - the Jacobi iteration converges within the maximum iterations,
 *   - the residual falls below epsilon,
 *   - the maximum absolute error against the exact solution is within
 *     a specified tolerance.
 *
 * Uses a simple pass/fail framework rather than Boost.Test, since
 * Boost.Test does not natively support MPI execution. Only rank 0
 * reports results.
 *
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <mpi.h>

#include "solver.hpp"
#include "test-case.hpp"
#include "halo-mpi.hpp"

/**
 * @brief Return the exact analytical solution for a verification case.
 */
double exact_solution(int test_case, double x, double y, double z) {
    switch (test_case) {
        case 1: 
            return x * x + y * y + z * z;
            
        case 2:
            return std::sin(M_PI * x) * std::sin(M_PI * y) * std::sin(M_PI * z);
            
        case 3:
            return std::sin(M_PI * x) * std::sin(4.0 * M_PI * y) * std::sin(8.0 * M_PI * z);
        
        default:
            return 0.0;
    }
}


/**
 * @brief Set the default grid dimensions for each verification case.
 */
void default_grid(int test_case, int& Nx, int& Ny, int& Nz) {
    switch (test_case) {
        case 1: 
            Nx = 32; Ny = 32;  Nz = 32; 
            break;
            
        case 2:
            Nx = 64; Ny = 64;  Nz = 64; 
            break;
        case 3: 
            Nx = 32; Ny = 64;  Nz = 128;
            break;
            
        default: 
            Nx = 32; Ny = 32; Nz = 32;  
            break;
    }
}


/**
 * @brief Run a verification case in parallel and check convergence and accuracy.
 *
 * Each rank participates in the solve. After convergence, the solution
 * is gathered to rank 0, which computes the L-infinity error against
 * the exact solution.
 *
 * @param test_case  Verification case identifier (1-3).
 * @param sol_tol    Maximum allowed L-infinity error.
 * @param rank       MPI rank of this process.
 * @param size       Total number of MPI processes.
 * @return true if the test passes on rank 0, true on all other ranks.
 */
bool run_test(int test_case, double sol_tol, int rank, int size) {

    int Nx, Ny, Nz;
    default_grid(test_case, Nx, Ny, Nz);

    double hx    = 1.0 / (Nx - 1.0);
    double hy    = 1.0 / (Ny - 1.0);
    double hz    = 1.0 / (Nz - 1.0);
    double a     = 1.0 / (hx * hx);
    double b     = 1.0 / (hy * hy);
    double c     = 1.0 / (hz * hz);
    double denom = 1.0 / (2.0 * (a + b + c));

    /// For simplicity, decompose only in z
    int Px = 1, Py = 1, Pz = size;

    int rx = rank % Px;
    int ry = (rank / Px) % Py;
    int rz = rank / (Px * Py);

    int loc_Nx = Nx / Px + (rx < Nx % Px ? 1 : 0);
    int loc_Ny = Ny / Py + (ry < Ny % Py ? 1 : 0);
    int loc_Nz = Nz / Pz + (rz < Nz % Pz ? 1 : 0);

    int start_x = rx * (Nx / Px) + std::min(rx, Nx % Px);
    int start_y = ry * (Ny / Py) + std::min(ry, Ny % Py);
    int start_z = rz * (Nz / Pz) + std::min(rz, Nz % Pz);

    int pNx = loc_Nx + 2;
    int pNy = loc_Ny + 2;
    int pNz = loc_Nz + 2;
    int pN  = pNx * pNy * pNz;

    /// Neighbour identification
    auto rank_from_coords = [&](int cx, int cy, int cz) -> int {
        if (cx < 0 || cx >= Px || cy < 0 || cy >= Py || cz < 0 || cz >= Pz)
            return -1;
        return cx + Px * (cy + Py * cz);
    };

    int nbr_xm = rank_from_coords(rx - 1, ry, rz);
    int nbr_xp = rank_from_coords(rx + 1, ry, rz);
    int nbr_ym = rank_from_coords(rx, ry - 1, rz);
    int nbr_yp = rank_from_coords(rx, ry + 1, rz);
    int nbr_zm = rank_from_coords(rx, ry, rz - 1);
    int nbr_zp = rank_from_coords(rx, ry, rz + 1);

    /// Allocate padded local arrays
    double* u     = new double[pN]();
    double* u_new = new double[pN]();
    double* f     = new double[pN]();

    /// Initialise local subdomain via temporary non-padded arrays
    int loc_N_noghost = loc_Nx * loc_Ny * loc_Nz;
    double* u_tmp     = new double[loc_N_noghost]();
    double* u_new_tmp = new double[loc_N_noghost]();
    double* f_tmp     = new double[loc_N_noghost]();

    initialise_test_case_local(test_case, u_tmp, u_new_tmp, f_tmp,
                               loc_Nx, loc_Ny, loc_Nz,
                               start_x, start_y, start_z,
                               Nx, Ny, Nz, hx, hy, hz);

    /// Copy into padded arrays
    for (int k = 0; k < loc_Nz; ++k) {
        for (int j = 0; j < loc_Ny; ++j) {
            for (int i = 0; i < loc_Nx; ++i) {
                int idx_src = i + loc_Nx * (j + loc_Ny * k);
                int idx_dst = (i + 1) + pNx * ((j + 1) + pNy * (k + 1));
                u[idx_dst]     = u_tmp[idx_src];
                u_new[idx_dst] = u_new_tmp[idx_src];
                f[idx_dst]     = f_tmp[idx_src];
            }
        }
    }
    delete[] u_tmp;
    delete[] u_new_tmp;
    delete[] f_tmp;

    /// Solve
    double epsilon  = 1e-8;
    double res_nrm  = 1.0;
    int    iter     = 0;
    int    max_iter = 100000;

    /// Interior loop bounds: skip global boundary points
    int i_start = (nbr_xm < 0) ? 2 : 1;
    int i_end   = (nbr_xp < 0) ? pNx - 3 : pNx - 2;
    int j_start = (nbr_ym < 0) ? 2 : 1;
    int j_end   = (nbr_yp < 0) ? pNy - 3 : pNy - 2;
    int k_start = (nbr_zm < 0) ? 2 : 1;
    int k_end   = (nbr_zp < 0) ? pNz - 3 : pNz - 2;

    while (res_nrm > epsilon && iter < max_iter) {
        halo_exchange(u, loc_Nx, loc_Ny, loc_Nz,
                      nbr_xm, nbr_xp, nbr_ym, nbr_yp, nbr_zm, nbr_zp);

        res_nrm = jacobi_residual(u, u_new, f, pNx, pNy, pNz,
                                  i_start, i_end, j_start, j_end, k_start, k_end,
                                  a, b, c, denom);
        ++iter;
    }

    /// Pack local real data (excluding ghosts)
    double* send_buf = new double[loc_N_noghost]();
    for (int k = 0; k < loc_Nz; ++k) {
        for (int j = 0; j < loc_Ny; ++j) {
            for (int i = 0; i < loc_Nx; ++i) {
                int idx_pad = (i + 1) + pNx * ((j + 1) + pNy * (k + 1));
                int idx_buf = i + loc_Nx * (j + loc_Ny * k);
                send_buf[idx_buf] = u[idx_pad];
            }
        }
    }

    /// Gather to rank 0
    double* u_global = nullptr;
    if (rank == 0) {
        u_global = new double[Nx * Ny * Nz]();
    }

    if (rank == 0) {
        /// Copy rank 0's data
        for (int k = 0; k < loc_Nz; ++k) {
            for (int j = 0; j < loc_Ny; ++j) {
                for (int i = 0; i < loc_Nx; ++i) {
                    int gi = start_x + i;
                    int gj = start_y + j;
                    int gk = start_z + k;
                    int global_idx = gi + Nx * (gj + Ny * gk);
                    int local_idx  = i + loc_Nx * (j + loc_Ny * k);
                    u_global[global_idx] = send_buf[local_idx];
                }
            }
        }

        /// Receive from other ranks
        for (int r = 1; r < size; ++r) {
            int r_rz = r;  // z-only decomposition
            int r_loc_Nz = Nz / Pz + (r_rz < Nz % Pz ? 1 : 0);
            int r_start_z = r_rz * (Nz / Pz) + std::min(r_rz, Nz % Pz);
            int r_loc_N = loc_Nx * loc_Ny * r_loc_Nz;

            double* recv_buf = new double[r_loc_N]();
            MPI_Recv(recv_buf, r_loc_N, MPI_DOUBLE, r, 30, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int k = 0; k < r_loc_Nz; ++k) {
                for (int j = 0; j < loc_Ny; ++j) {
                    for (int i = 0; i < loc_Nx; ++i) {
                        int gi = i;
                        int gj = j;
                        int gk = r_start_z + k;
                        int global_idx = gi + Nx * (gj + Ny * gk);
                        int local_idx  = i + loc_Nx * (j + loc_Ny * k);
                        u_global[global_idx] = recv_buf[local_idx];
                    }
                }
            }
            delete[] recv_buf;
        }
    } else {
        MPI_Send(send_buf, loc_N_noghost, MPI_DOUBLE, 0, 30, MPI_COMM_WORLD);
    }

    delete[] send_buf;
    delete[] u;
    delete[] u_new;
    delete[] f;

    /// Rank 0 checks results
    bool pass = true;
    if (rank == 0) {
        /// Check convergence
        if (iter >= max_iter) {
            std::cout << "  FAILED: did not converge within " << max_iter << " iterations" << std::endl;
            pass = false;
        }
        if (res_nrm >= epsilon) {
            std::cout << "  FAILED: residual " << res_nrm << " >= " << epsilon << std::endl;
            pass = false;
        }

        /// Check accuracy
        double max_err = 0.0;
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = i + Nx * (j + Ny * k);
                    double x = i * hx;
                    double y = j * hy;
                    double z = k * hz;
                    double err = std::fabs(u_global[idx] - exact_solution(test_case, x, y, z));
                    if (err > max_err) max_err = err;
                }
            }
        }

        std::cout << "  Residual = " << res_nrm
                  << ", Iterations = " << iter
                  << ", Max error = " << max_err << std::endl;

        if (max_err >= sol_tol) {
            std::cout << "  FAILED: max error " << max_err << " >= " << sol_tol << std::endl;
            pass = false;
        }

        delete[] u_global;
    }

    return pass;
}


/**
 * @brief Run all MPI verification tests and report results.
 */
int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int failures = 0;

    if (rank == 0) std::cout << "=== MPI Verification Case 1: Polynomial (32^3) ===" << std::endl;
    if (!run_test(1, 1e-4, rank, size)) ++failures;
    if (rank == 0) std::cout << (failures == 0 ? "  PASSED" : "") << std::endl;

    int prev_failures = failures;
    if (rank == 0) std::cout << "=== MPI Verification Case 2: Smooth sinusoidal (64^3) ===" << std::endl;
    if (!run_test(2, 1e-3, rank, size)) ++failures;
    if (rank == 0 && failures == prev_failures) std::cout << "  PASSED" << std::endl;

    prev_failures = failures;
    if (rank == 0) std::cout << "=== MPI Verification Case 3: Anisotropic oscillatory (32x64x128) ===" << std::endl;
    if (!run_test(3, 1e-2, rank, size)) ++failures;
    if (rank == 0 && failures == prev_failures) std::cout << "  PASSED" << std::endl;

    if (rank == 0) {
        std::cout << std::endl;
        if (failures == 0) {
            std::cout << "All MPI verification tests PASSED." << std::endl;
        } else {
            std::cout << failures << " MPI test(s) FAILED." << std::endl;
        }
    }

    MPI_Finalize();
    return (failures == 0) ? 0 : 1;
}