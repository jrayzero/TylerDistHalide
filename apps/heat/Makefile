include ../support/Makefile.inc

W ?=
H ?=
D ?= 

all: distributed_heat

distributed_heat: heat_distributed.cpp
	$(CXX) $(CXXFLAGS) $(MPI_CXX_FLAGS) -O3 heat_distributed.cpp $(LIB_HALIDE) -o distributed_heat -ldl -lpthread -lz $(LDFLAGS)

run_distributed_heat: distributed_heat
	$(MPI_RUN) ./distributed_heat $(W) $(H) $(D)

clean:
	rm -f distributed_heat
