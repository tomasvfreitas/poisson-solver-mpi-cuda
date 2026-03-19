/**
 * @file solver-cuda.cu
 * @brief CUDA kernel for the fused Jacobi sweep and residual computation.
 */


#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include "solver.hpp"


/**
 * @brief CUDA kernel: fused Jacobi update, residual accumulation and copy-back.
 *
 * Each thread processes one interior grid point. The residual contribution
 * is accumulated into a global sum using atomicAdd.
 */
__global__
void jacobi_residual_kernel(double* u, double* u_new, const double* f,
                            int Nx, int Ny, int Nz,
                            int i_start, int i_end,
                            int j_start, int j_end,
                            int k_start, int k_end,
                            double a, double b, double c, double denom,
                            double* d_res_sum) {

    int i = i_start + blockIdx.x * blockDim.x + threadIdx.x;
    int j = j_start + blockIdx.y * blockDim.y + threadIdx.y;
    int k = k_start + blockIdx.z * blockDim.z + threadIdx.z;

    if (i <= i_end && j <= j_end && k <= k_end) {

        int idx = i + Nx * (j + Ny * k);
        int NxNy = Nx * Ny;

        // Jacobi update
        u_new[idx] = denom * (
            (u[idx + 1]    + u[idx - 1])    * a +
            (u[idx + Nx]   + u[idx - Nx])   * b +
            (u[idx + NxNy] + u[idx - NxNy]) * c -
            f[idx]);

        // Residual on current solution
        double lap =
            (u[idx + 1]    - 2.0 * u[idx] + u[idx - 1])    * a +
            (u[idx + Nx]   - 2.0 * u[idx] + u[idx - Nx])   * b +
            (u[idx + NxNy] - 2.0 * u[idx] + u[idx - NxNy]) * c;
        double r = f[idx] - lap;
        atomicAdd(d_res_sum, r * r);

        // Copy back
        u[idx] = u_new[idx];
    }
}

/**
 * @brief Host wrapper: launches the CUDA kernel and returns the global residual norm.
 *
 * Allocates a device scalar for the residual sum, launches the kernel
 * with a 3D grid of threads, retrieves the result and combines it
 * across MPI ranks using MPI_Allreduce.
 */
double jacobi_residual_cuda(double* d_u, double* d_u_new, const double* d_f,
                            int Nx, int Ny, int Nz,
                            int i_start, int i_end,
                            int j_start, int j_end,
                            int k_start, int k_end,
                            double a, double b, double c, double denom) {

    /// Allocate and zero the residual sum on device
    double* d_res_sum;
    cudaMalloc(&d_res_sum, sizeof(double));
    cudaMemset(d_res_sum, 0, sizeof(double));

    /// Compute interior dimensions
    int ni = i_end - i_start + 1;
    int nj = j_end - j_start + 1;
    int nk = k_end - k_start + 1;

    /// Configure 3D grid of threads (512 threads per block)
    dim3 threads(8, 8, 8);
    dim3 blocks((ni + threads.x - 1) / threads.x,
                (nj + threads.y - 1) / threads.y,
                (nk + threads.z - 1) / threads.z);

    /// Launch kernel
    jacobi_residual_kernel<<<blocks, threads>>>(
        d_u, d_u_new, d_f, Nx, Ny, Nz,
        i_start, i_end, j_start, j_end, k_start, k_end,
        a, b, c, denom, d_res_sum);
        
    /// Synchronise to make sure all threads have finished executing
    cudaDeviceSynchronize();

    /// Copy residual sum back to host
    double local_sum = 0.0;
    cudaMemcpy(&local_sum, d_res_sum, sizeof(double), cudaMemcpyDeviceToHost);
    
    /// Free GPU memory
    cudaFree(d_res_sum);

    /// Combine across MPI ranks
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(global_sum);
}