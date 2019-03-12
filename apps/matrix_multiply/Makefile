include ../support/Makefile.inc

W ?=
H ?=

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

distributed_matrix_multiply: matrix_multiply_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 matrix_multiply_distributed.cpp $(LIB_HALIDE) -o distributed_matrix_multiply -ldl -lpthread -lz $(LDFLAGS)

run_distributed_matrix_multiply: distributed_matrix_multiply
	$(MPI_RUN) ./distributed_matrix_multiply $(W) $(H)

clean:
	rm -f test distributed_matrix_multiply
