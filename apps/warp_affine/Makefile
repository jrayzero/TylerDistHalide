include ../support/Makefile.inc

W ?=
H ?=

all: test

halide_warp_affine: halide_warp_affine.cpp
	$(CXX) $(CXXFLAGS) halide_warp_affine.cpp $(LIB_HALIDE) -o halide_warp_affine -ldl -lpthread -lz $(LDFLAGS)

halide_warp_affine.o: halide_warp_affine
	./halide_warp_affine

# g++ on OS X might actually be system clang without openmp
CXX_VERSION=$(shell $(CXX) --version)
ifeq (,$(findstring clang,$(CXX_VERSION)))
OPENMP_FLAGS=-fopenmp
else
OPENMP_FLAGS=
endif

# -O2 is faster than -O3 for this app (O3 unrolls too much)
test: test.cpp halide_warp_affine.o
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -msse2 -Wall -O2 test.cpp halide_warp_affine.o -o test -lpthread -ldl $(PNGFLAGS)

distributed_warp_affine: halide_warp_affine_distributed.cpp
	$(CXX) $(CXXFLAGS) $(MPI_CXX_FLAGS) -O3 halide_warp_affine_distributed.cpp $(LIB_HALIDE) -o distributed_warp_affine -ldl -lpthread -lz $(LDFLAGS)

run_distributed_warp_affine: distributed_warp_affine
	$(MPI_RUN) --oversubscribe ./distributed_warp_affine $(W) $(H)

nondistributed_warp_affine: ../../ halide_warp_affine_distributed.cpp
	$(MAKE) -C ../../ $(LIB_HALIDE)
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 halide_warp_affine_distributed.cpp -DNON_DISTRIBUTED ../../$(LIB_HALIDE) -o nondistributed_warp_affine -ldl -lpthread -lz $(LDFLAGS)

clean:
	rm -f test halide_warp_affine.o halide_warp_affine distributed_warp_affine
