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

distributed_transpose: transpose_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 transpose_distributed.cpp $(LIB_HALIDE) -o distributed_transpose -ldl -lpthread -lz $(LDFLAGS)

run_distributed_transpose: distributed_transpose
	$(MPI_RUN) ./distributed_transpose $(W) $(H)

clean:
	rm -f test distributed_transpose
