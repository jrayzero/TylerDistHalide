include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), $(MPI_BUILD_DIR)/bin/mpic++, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )
MPI_RUN=$(MPI_BUILD_DIR)/bin/mpirun -x LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/tmp/ -np $(NUM_MPI_RANKS) -host $(MPI_NODES) --map-by node --mca btl_openib_allow_ib 1

all: distributed_sobel

distributed_sobel: sobel_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 sobel_distributed.cpp $(LIB_HALIDE) -o distributed_sobel -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS)

run_distributed_sobel: distributed_sobel
	$(MPI_RUN) ./distributed_sobel

clean:
	rm -f distributed_sobel
