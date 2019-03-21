include ../support/Makefile.inc

W ?=
H ?=

all: test

halide_convolution: halide_convolution.cpp
	$(CXX) $(CXXFLAGS) halide_convolution.cpp $(LIB_HALIDE) -o halide_convolution -ldl -lpthread -lz $(LDFLAGS)

halide_convolution.o: halide_convolution
	./halide_convolution

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

# -O2 is faster than -O3 for this app (O3 unrolls too much)
test: test.cpp halide_convolution.o
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -msse2 -Wall -O2 test.cpp halide_convolution.o -o test -lpthread -ldl $(PNGFLAGS)

distributed_convolution: halide_convolution_distributed.cpp
	$(CXX) $(CXXFLAGS) $(MPI_CXX_FLAGS) -O3 halide_convolution_distributed.cpp $(LIB_HALIDE) -o distributed_convolution -ldl -lpthread -lz $(LDFLAGS)

run_distributed_convolution: distributed_convolution
	$(MPI_RUN) ./distributed_convolution $(W) $(H)

nondistributed_convolution: ../../ halide_convolution_distributed.cpp
	$(MAKE) -C ../../ $(LIB_HALIDE)
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 halide_convolution_distributed.cpp -DNON_DISTRIBUTED ../../$(LIB_HALIDE) -o nondistributed_convolution -ldl -lpthread -lz $(LDFLAGS)

clean:
	rm -f test halide_convolution.o halide_convolution distributed_convolution
