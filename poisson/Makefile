CXX        = g++
CXXFLAGS   = -std=c++11 -Wall -O2
LDLIBS     = -lblas
BOOST_LIBS = -lboost_program_options

SERIAL_OBJ = main-serial.o solver-serial.o test-case.o io.o
TEST_OBJ   = test-serial.o solver-serial.o test-case.o
HEADERS    = solver.hpp test-case.hpp io.hpp

default: all
all: poisson

poisson: $(SERIAL_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS) $(BOOST_LIBS)

tests: $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)
	./$@

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

doc:
	doxygen Doxyfile


.PHONY: default all tests doc clean
clean:
	rm -f *.o poisson tests solution.txt