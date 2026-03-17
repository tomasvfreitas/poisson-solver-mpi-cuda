/**
 * @file main-mpi.cpp
 * @brief Parallel Jacobi solver for the 3D Poisson equation.
 *
 * This program solves the Poisson equation \f$ \nabla^2 u = f \f$ on 
 * a three-dimensional grid using Jacobi iteration with Dirichlet boundary conditions.
 *
 * The solution is iteratively updated using a finite-difference discretisation 
 * of the Laplacian operator until the L2 norm of the residual \f$ \|r\|_2 \f$
 * falls below a specified tolerance.
 *
 * The implementation uses the BLAS routine 'dcopy' to update the
 * solution array after each Jacobi iteration.
 *
 * @author Tomas Freitas
 * @date 2026
 */

#include <iostream>
#include <cmath>
#include <string>
#include <boost/program_options.hpp>
#include <mpi.h>

#include "solver.hpp"
#include "test-case.hpp"
#include "io.hpp"
#include "halo-mpi.hpp"

namespace po = boost::program_options;

#define F77NAME(x) x##_
extern "C" {
    void F77NAME(dcopy)(const int& N,
                        const double* x, const int& incx,
                        double* y, const int& incy);
}

int main(int argc, char* argv[]) {
    
    MPI_Init(&argc, &argv);                   ///< Initialising MPI
    
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);     ///< Obtaining rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);     ///< Obatining number of processes

    /// Command-line parsing
    po::options_description opts("Allowed options");
    opts.add_options()
        ("help", "Print available options.")
        ("forcing", po::value<std::string>(), "Input forcing file")
        ("test", po::value<int>()->default_value(1)->implicit_value(1), "Test case to use (1-5)")
        ("Nx", po::value<int>()->default_value(32), "Number of grid points (x)")
        ("Ny", po::value<int>()->default_value(32), "Number of grid points (y)")
        ("Nz", po::value<int>()->default_value(32), "Number of grid points (z)")
        ("epsilon", po::value<double>()->default_value(1e-8), "Residual threshold")
        ("Px", po::value<int>()->default_value(1), "Number of processes (x)")
        ("Py", po::value<int>()->default_value(1), "Number of processes (y)")
        ("Pz", po::value<int>()->default_value(1), "Number of processes (z)");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);
    
    /// If user provides option help, print options and exit
    if (vm.count("help")) {
        if (rank == 0) {
            std::cout << opts << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    
    /// Check if user input --test and/or --forcing file, if neither or both were selected and give error
    bool has_test    = vm.count("test") && !vm["test"].defaulted();
    bool has_forcing = vm.count("forcing");
    if (has_test == has_forcing) {
        if (rank == 0) {
            std::cerr << "Error: Exactly one of --test or --forcing must be provided." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /// Check if given a forcing function the user also specified grid points and give error
    bool Nx_given = vm.count("Nx") && !vm["Nx"].defaulted();
    bool Ny_given = vm.count("Ny") && !vm["Ny"].defaulted();
    bool Nz_given = vm.count("Nz") && !vm["Nz"].defaulted();
    if (has_forcing && (Nx_given || Ny_given || Nz_given)) {
        if (rank == 0) {
            std::cerr << "Error: --Nx, --Ny and --Nz may only be specified with --test." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /// Checking for valid test number
    int  test_case   = vm["test"].as<int>();
    if (has_test && (test_case < 1 || test_case > 5)) {
        if (rank == 0) {
            std::cerr << "Error: test case must be between 1 and 5." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    /// Check if Px * Py * Pz = number of processes, if not, program aborts
    int Px = vm["Px"].as<int>();              ///< Number of processes in x
    int Py = vm["Py"].as<int>();              ///< Number of processes in y
    int Pz = vm["Pz"].as<int>();              ///< Number of processes in z
    int P  = Px * Py * Pz;                    ///< Total number of processes
    if (P != size) {
        if (rank == 0) {
            std::cerr << "Error: Number of MPI ranks must be equal to Px * Py * Pz" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int Nx, Ny, Nz;                           ///< Initialising grid dimension
    double* f_global = nullptr;               ///< Initialising forcing function
    
    /* If --forcing then rank 0 reads forcing file and then Broadcast Nx, Ny, Nz for all ranks
     * If --test then read inputs of  Nx, Ny and Nz in all ranks
     */
    if (has_forcing) {
        if (rank == 0) {
            read_forcing(vm["forcing"].as<std::string>(), f_global, Nx, Ny, Nz);
        }
        MPI_Bcast(&Nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&Ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&Nz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else {
        Nx = vm["Nx"].as<int>();              ///< Number of grid points in x-direction
        Ny = vm["Ny"].as<int>();              ///< Number of grid points in y-direction
        Nz = vm["Nz"].as<int>();              ///< Number of grid points in z-direction
    }
    
    double hx    = 1.0 / (Nx - 1.0);          ///< Grid spacing in x-direction
    double hy    = 1.0 / (Ny - 1.0);          ///< Grid spacing in y-direction
    double hz    = 1.0 / (Nz - 1.0);          ///< Grid spacing in z-direction

    double a     = 1.0 / (hx * hx);           ///< Finite-difference coefficient in x-direction
    double b     = 1.0 / (hy * hy);           ///< Finite-difference coefficient in y-direction
    double c     = 1.0 / (hz * hz);           ///< Finite-difference coefficient in z-direction
    double denom = 1.0 / (2.0 * (a + b + c)); ///< Setting denom for simplicity
    
    
    int rx = rank % Px;                       ///< Process x-coordinates
    int ry = (rank / Px) % Py;                ///< Process y-coordinates
    int rz = rank / (Px * Py);                ///< Process z-coordinates
    
    /// Local grid sizes
    int loc_Nx = Nx / Px + (rx < Nx % Px ? 1 : 0);
    int loc_Ny = Ny / Py + (ry < Ny % Py ? 1 : 0);
    int loc_Nz = Nz / Pz + (rz < Nz % Pz ? 1 : 0);
    
    /// Global starting indices
    int start_x = rx * (Nx / Px) + std::min(rx, Nx % Px);
    int start_y = ry * (Ny / Py) + std::min(ry, Ny % Py);
    int start_z = rz * (Nz / Pz) + std::min(rz, Nz % Pz);
    
    /// Padded dimensions (including ghost layers)
    int pNx = loc_Nx + 2;
    int pNy = loc_Ny + 2;
    int pNz = loc_Nz + 2;
    int pN  = pNx * pNy * pNz;

    // Neighbour identification
    /// Helper to convert process coordinates to rank (-1 if out of bounds)
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

    /// Allocate local arrays (with ghost layers)
    double* u     = new double[pN]();
    double* u_new = new double[pN]();
    double* f     = new double[pN]();

    /// Initialise local subdomain
    if (has_test) {
        /// Initialise into a temporary array without ghost padding,
        /// then copy into the padded array at the correct offset.
        int loc_N_noghost = loc_Nx * loc_Ny * loc_Nz;
        double* u_tmp     = new double[loc_N_noghost]();
        double* u_new_tmp = new double[loc_N_noghost]();
        double* f_tmp     = new double[loc_N_noghost]();

        initialise_test_case_local(test_case, u_tmp, u_new_tmp, f_tmp,
                                   loc_Nx, loc_Ny, loc_Nz,
                                   start_x, start_y, start_z,
                                   Nx, Ny, Nz, hx, hy, hz);

        /// Copy from non-padded to padded arrays (offset by 1 in each direction)
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

    } else if (has_forcing && rank == 0) {
        /// Distribute forcing data from rank 0 to all ranks
        /// For simplicity, rank 0 sends each rank's portion
        for (int r = 0; r < size; ++r) {
            int r_rx = r % Px;
            int r_ry = (r / Px) % Py;
            int r_rz = r / (Px * Py);

            int r_loc_Nx = Nx / Px + (r_rx < Nx % Px ? 1 : 0);
            int r_loc_Ny = Ny / Py + (r_ry < Ny % Py ? 1 : 0);
            int r_loc_Nz = Nz / Pz + (r_rz < Nz % Pz ? 1 : 0);

            int r_start_x = r_rx * (Nx / Px) + std::min(r_rx, Nx % Px);
            int r_start_y = r_ry * (Ny / Py) + std::min(r_ry, Ny % Py);
            int r_start_z = r_rz * (Nz / Pz) + std::min(r_rz, Nz % Pz);

            /// Pack this rank's portion of the global forcing into a buffer
            int r_loc_N = r_loc_Nx * r_loc_Ny * r_loc_Nz;
            double* buf = new double[r_loc_N]();

            for (int k = 0; k < r_loc_Nz; ++k) {
                for (int j = 0; j < r_loc_Ny; ++j) {
                    for (int i = 0; i < r_loc_Nx; ++i) {
                        int gi = r_start_x + i;
                        int gj = r_start_y + j;
                        int gk = r_start_z + k;
                        int global_idx = gi + Nx * (gj + Ny * gk);
                        int local_idx  = i + r_loc_Nx * (j + r_loc_Ny * k);
                        buf[local_idx] = f_global[global_idx];
                    }
                }
            }

            if (r == 0) {
                /// Copy directly into rank 0's padded array
                for (int k = 0; k < r_loc_Nz; ++k) {
                    for (int j = 0; j < r_loc_Ny; ++j) {
                        for (int i = 0; i < r_loc_Nx; ++i) {
                            int idx_src = i + r_loc_Nx * (j + r_loc_Ny * k);
                            int idx_dst = (i + 1) + pNx * ((j + 1) + pNy * (k + 1));
                            f[idx_dst] = buf[idx_src];
                        }
                    }
                }
            } else {
                MPI_Send(buf, r_loc_N, MPI_DOUBLE, r, 10, MPI_COMM_WORLD);
            }
            delete[] buf;
        }
        delete[] f_global;

    } else if (has_forcing && rank != 0) {
        /// Receive forcing data from rank 0
        int loc_N_noghost = loc_Nx * loc_Ny * loc_Nz;
        double* buf = new double[loc_N_noghost]();
        MPI_Recv(buf, loc_N_noghost, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int k = 0; k < loc_Nz; ++k) {
            for (int j = 0; j < loc_Ny; ++j) {
                for (int i = 0; i < loc_Nx; ++i) {
                    int idx_src = i + loc_Nx * (j + loc_Ny * k);
                    int idx_dst = (i + 1) + pNx * ((j + 1) + pNy * (k + 1));
                    f[idx_dst] = buf[idx_src];
                }
            }
        }
        delete[] buf;
    }

    // Solver loop

    double epsilon       = vm["epsilon"].as<double>();
    double residual_nrm2 = 1.0;
    int    iter          = 0;
    int    max_iter      = 100000;

    while (residual_nrm2 > epsilon && iter < max_iter) {
        /// Exchange ghost layers with neighbours
        halo_exchange(u, loc_Nx, loc_Ny, loc_Nz,
                      nbr_xm, nbr_xp, nbr_ym, nbr_yp, nbr_zm, nbr_zp);

        /// Jacobi sweep on padded local grid (interior points are [1..loc_Nx] etc.)
        jacobi(u, u_new, f, pNx, pNy, pNz, a, b, c, denom);

        /// Copy u_new to u
        F77NAME(dcopy)(pN, u_new, 1, u, 1);

        /// Compute global residual
        residual_nrm2 = residual(u, f, pNx, pNy, pNz, a, b, c);

        ++iter;
    }

    // Gather solution and write output

    /// Allocate global solution on rank 0
    double* u_global = nullptr;
    if (rank == 0) {
        u_global = new double[Nx * Ny * Nz]();
    }

    /// Pack local real data (excluding ghosts) into a buffer
    int loc_N_noghost = loc_Nx * loc_Ny * loc_Nz;
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

    if (rank == 0) {
        /// Copy rank 0's data into the global array
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

        /// Receive from all other ranks
        for (int r = 1; r < size; ++r) {
            int r_rx = r % Px;
            int r_ry = (r / Px) % Py;
            int r_rz = r / (Px * Py);

            int r_loc_Nx = Nx / Px + (r_rx < Nx % Px ? 1 : 0);
            int r_loc_Ny = Ny / Py + (r_ry < Ny % Py ? 1 : 0);
            int r_loc_Nz = Nz / Pz + (r_rz < Nz % Pz ? 1 : 0);

            int r_start_x = r_rx * (Nx / Px) + std::min(r_rx, Nx % Px);
            int r_start_y = r_ry * (Ny / Py) + std::min(r_ry, Ny % Py);
            int r_start_z = r_rz * (Nz / Pz) + std::min(r_rz, Nz % Pz);

            int r_loc_N = r_loc_Nx * r_loc_Ny * r_loc_Nz;
            double* recv_buf = new double[r_loc_N]();
            MPI_Recv(recv_buf, r_loc_N, MPI_DOUBLE, r, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int k = 0; k < r_loc_Nz; ++k) {
                for (int j = 0; j < r_loc_Ny; ++j) {
                    for (int i = 0; i < r_loc_Nx; ++i) {
                        int gi = r_start_x + i;
                        int gj = r_start_y + j;
                        int gk = r_start_z + k;
                        int global_idx = gi + Nx * (gj + Ny * gk);
                        int local_idx  = i + r_loc_Nx * (j + r_loc_Ny * k);
                        u_global[global_idx] = recv_buf[local_idx];
                    }
                }
            }
            delete[] recv_buf;
        }

        std::cout << "Final residual = " << residual_nrm2 << std::endl;
        write_solution("solution.txt", u_global, Nx, Ny, Nz);
        delete[] u_global;

    } else {
        MPI_Send(send_buf, loc_N_noghost, MPI_DOUBLE, 0, 20, MPI_COMM_WORLD);
    }

    delete[] send_buf;

    // ── Cleanup ───────────────────────────────────────────────────────

    delete[] u;
    delete[] u_new;
    delete[] f;

    MPI_Finalize();
    return 0;
}