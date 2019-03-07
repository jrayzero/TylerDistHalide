include ../support/Makefile.inc

WITH_MPI ?=
CXX := $(if $(WITH_MPI), $(MPI_BUILD_DIR)/bin/mpic++, $(CXX))
MPI_CXX_FLAGS=$(if $(WITH_MPI), -DWITH_MPI=1 -DDUMP_RESULTS=$(DUMP_RESULTS), )
MPI_RUN=$(MPI_BUILD_DIR)/bin/mpirun -x LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):/tmp/ -np $(NUM_MPI_RANKS) -host $(MPI_NODES) --map-by node --mca btl_openib_allow_ib 1

all: distributed_heat

distributed_heat: heat_distributed.cpp
	$(CXX) $(CXXFLAGS) $(MPI_CXX_FLAGS) -O3 heat_distributed.cpp $(LIB_HALIDE) -o distributed_heat -ldl -lpthread -lz $(LDFLAGS)

run_distributed_heat: distributed_heat
	$(MPI_RUN) ./distributed_heat

clean:
	rm -f distributed_heat
