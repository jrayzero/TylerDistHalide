include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), $(MPI_BUILD_DIR)/bin/mpic++, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1 -DDUMP_RESULTS=$(DUMP_RESULTS), )
MPI_RUN=$(MPI_BUILD_DIR)/bin/mpirun -x LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/tmp/ -np $(NUM_MPI_RANKS) -host $(MPI_NODES) --map-by node --mca btl_openib_allow_ib 1

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
	$(MPI_RUN) ./distributed_matrix_multiply

clean:
	rm -f test distributed_matrix_multiply
