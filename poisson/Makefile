# MPI paths obtained from: mpicxx -show (on typhoon.ae.ic.ac.uk)
MPI_INC = -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB = -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi

CXX          = g++
MPICXX       = mpicxx
NVCC         = nvcc
CXXFLAGS     = -std=c++11 -Wall -O3
NVCCFLAGS    = -O3 -DCUDA_ENABLED $(MPI_INC)
LDLIBS       = -lblas
BOOST_LIBS   = -lboost_program_options

SERIAL_OBJ   = main-serial.o solver-serial.o test-case.o io.o
MPI_OBJ      = main-mpi.o solver-mpi.o test-case.o io.o halo-mpi.o
MPI_OBJ_CUDA = main-mpi-cuda.o solver-mpi-cuda.o test-case.o io.o halo-mpi.o solver-cuda.o
TEST_SER_OBJ = test-serial.o solver-serial.o test-case.o
TEST_MPI_OBJ = test-mpi.o solver-mpi.o test-case.o halo-mpi.o
HEADERS      = solver.hpp test-case.hpp io.hpp halo-mpi.hpp


default: all
all: poisson

poisson: $(SERIAL_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS) $(BOOST_LIBS)

poisson-mpi: $(MPI_OBJ)
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(BOOST_LIBS)

poisson-cuda: $(MPI_OBJ_CUDA)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(BOOST_LIBS) $(MPI_INC) $(MPI_LIB)


tests: test-serial test-mpi
	./test-serial
	mpiexec -np 4 ./test-mpi

test-serial: $(TEST_SER_OBJ) 
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

test-mpi: $(TEST_MPI_OBJ)
	$(MPICXX) $(CXXFLAGS) -o $@ $^


%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%-mpi.o: %-mpi.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

%-cuda.o: %-cuda.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%-mpi-cuda.o: %-mpi.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@


doc:
	doxygen Doxyfile


.PHONY: default all tests doc clean
clean:
	rm -f *.o poisson poisson-mpi poisson-cuda test-serial test-mpi solution.txt