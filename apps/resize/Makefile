include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), $(MPI_BUILD_DIR)/bin/mpic++, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1, )
MPI_RUN=$(MPI_BUILD_DIR)/bin/mpirun -x LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/tmp/ -np $(NUM_MPI_RANKS) -host $(MPI_NODES) --map-by node --mca btl_openib_allow_ib 1

CXXFLAGS += -Wall -march=corei7-avx -mtune=corei7-avx

.PHONY: clean

resize: ../../ resize.cpp
	$(CXX) $(CXXFLAGS) resize.cpp $(LIB_HALIDE) -o resize -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_resize: resize_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) resize_distributed.cpp $(LIB_HALIDE) -o distributed_resize -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

run_distributed_resize: distributed_resize
	$(MPI_RUN) ./distributed_resize

distributed_downsample: downsample_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) downsample_distributed.cpp $(LIB_HALIDE) -o distributed_downsample -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

distributed_upsample: upsample_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) upsample_distributed.cpp $(LIB_HALIDE) -o distributed_upsample -lpthread -ldl -lz $(PNGFLAGS) $(LDFLAGS)

out.png: resize
	./resize ../images/rgba.png out.png -f 2.0 -t cubic -s 3

clean:
	rm -f out.png resize
