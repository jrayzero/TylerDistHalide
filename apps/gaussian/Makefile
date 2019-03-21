include ../support/Makefile.inc

W ?=
H ?=

all: test

halide_gaussian: halide_gaussian.cpp
	$(CXX) $(CXXFLAGS) halide_gaussian.cpp $(LIB_HALIDE) -o halide_gaussian -ldl -lpthread -lz $(LDFLAGS)

halide_gaussian.o: halide_gaussian
	./halide_gaussian

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

distributed_gaussian: halide_gaussian_distributed.cpp
	$(CXX) $(CXXFLAGS) $(MPI_CXX_FLAGS) -O3 halide_gaussian_distributed.cpp $(LIB_HALIDE) -o distributed_gaussian -ldl -lpthread -lz $(LDFLAGS)

run_distributed_gaussian: distributed_gaussian
	$(MPI_RUN) ./distributed_gaussian $(W) $(H)

clean:
	rm -f test halide_gaussian.o halide_gaussian distributed_gaussian
