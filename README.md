# Parallel 3D Poisson Solver — MPI + CUDA

A Jacobi iterative solver for the 3D Poisson equation ∇²u = f on the unit cube, parallelised
across a 3D Cartesian process grid with MPI and offloaded to GPU with CUDA.

Second-order central-difference discretisation, Dirichlet boundary conditions, iterating until
the L2 norm of the residual falls below tolerance.

---

## Results

**Optimisation** — cumulative effect of each step, measured with Intel VTune on a single rank:

| Step | CPU time (s) | Reduction |
|---|---|---|
| Baseline (`-O2`, BLAS `dcopy`, separate loops) | 1.710 | — |
| `-O2` → `-O3` | 1.350 | 21.1% |
| Replace `dcopy` with a plain loop | 1.180 | 12.6% |
| Fuse Jacobi sweep + copy + residual into one pass | 1.090 | 7.6% |
| **Total** | **1.090** | **36.3%** |

The `dcopy` result was counter-intuitive: at `-O3` the compiler vectorises a plain loop into a
memory copy that beats the BLAS routine, whose function-call overhead and internal threading cost
more than they save at this problem size.

**Strong scaling** — 64³ grid, single node:

| Cores | Wall time (s) | Speed-up | Efficiency |
|---|---|---|---|
| 1 | 13.579 | 1.00× | 100.0% |
| 8 | 1.914 | 7.09× | 88.7% |
| 16 | 1.063 | 12.78× | 79.9% |
| 48 | 0.589 | 23.05× | 48.0% |

Speed-up at 2 and 4 cores exceeds ideal (2.22× and 4.12×) because the per-rank working set starts
fitting in L3 cache. Beyond 8 cores, halo-exchange communication grows relative to the shrinking
per-rank computation.

**CUDA** — 3.9× per-iteration speed-up at 64³, 2.0× total wall time. The gap is because the GPU
needs roughly twice the iterations to converge under parallel residual evaluation.

---

## Build

Requires a C++11 compiler, BLAS, Boost.Program_options, Boost.Test and an MPI implementation.
The CUDA target additionally requires `nvcc`. The MPI include and library paths at the top of the
`Makefile` are set for the department cluster; update them from `mpicxx -show` on another machine.

```bash
cd poisson
make poisson        # serial
make poisson-mpi    # MPI
make poisson-cuda   # MPI + CUDA
make tests          # build and run both test suites
make doc            # Doxygen documentation
```

## Run

Either supply a forcing file or use one of the five built-in test cases. The solution is written
to `solution.txt`.

```bash
# Serial
./poisson --forcing <file>

# MPI: Px * Py * Pz must equal the rank count; keep the decomposition balanced
mpiexec -np 48 ./poisson-mpi --test 4 --Nx 64 --Ny 64 --Nz 64 --Px 4 --Py 4 --Pz 3

# CUDA: a single rank performs best
mpiexec -np 1 ./poisson-cuda --forcing <file> --Px 1 --Py 1 --Pz 1
```

`job-script.slr` is the SLURM script used for the strong and weak scaling studies.

## Test

```bash
./test-serial
mpiexec -np 4 ./test-mpi
```

Both suites run verification cases 1–3 and check convergence, the final residual, and the maximum
error against the analytical solution. The serial suite uses Boost.Test with separate convergence
and accuracy test cases; the MPI suite runs the same cases under the parallel decomposition with a
lightweight pass/fail harness, since Boost.Test does not support MPI execution natively.

---

## Design notes

**Domain decomposition.** The global grid is partitioned across a Px × Py × Pz process grid.
A 3D decomposition was chosen over slab or pencil alternatives because it minimises the
surface-to-volume ratio: communication volume per rank scales as O(N^(2/3)/P^(2/3)) rather than
O(N^(2/3)/P^(1/3)). Uneven divisions distribute the remainder one point each to the first ranks
in that direction.

**Halo exchange.** Each rank exchanges its outermost layer with its six neighbours via
`MPI_Sendrecv`, which avoids deadlock by combining send and receive in one call. Face data is
non-contiguous in memory, so MPI derived datatypes describe the layout directly rather than
packing into a buffer — avoiding two extra copies per exchange. The z and y faces use
`MPI_Type_vector`; the x face is non-contiguous in both y and z, so it uses a nested
`MPI_Type_vector` inside an `MPI_Type_create_hvector`. Ranks on the physical boundary pass
`MPI_PROC_NULL` so MPI skips the communication.

**Global residual.** One `MPI_Allreduce` per iteration rather than `MPI_Reduce` + `MPI_Bcast`,
halving the number of collectives so every rank can independently decide whether to continue.

**CUDA.** The fused `jacobi_residual()` kernel maps each interior point to a thread in 8×8×8
blocks. Unified managed memory keeps integration with the CPU-side halo exchange simple, at the
cost of per-iteration migration between host and device.

## Known limitations

- Halo exchange is blocking; non-blocking `MPI_Isend`/`MPI_Irecv` would allow interior computation
  to overlap with communication.
- The CUDA residual uses `atomicAdd`, which serialises thread updates. A shared-memory or
  warp-shuffle reduction would remove this bottleneck.
- Unified managed memory is costly across multiple ranks — explicit transfers for halo data only
  would be substantially faster.

---

Coursework for AERO70011 High Performance Computing, Imperial College London (2026).
