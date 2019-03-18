include ../support/Makefile.inc

W ?=
H ?=

all: distributed_cvtcolor

distributed_cvtcolor: cvtcolor_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 cvtcolor_distributed.cpp $(LIB_HALIDE) -o distributed_cvtcolor -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS)

run_distributed_cvtcolor: distributed_cvtcolor
	$(MPI_RUN) ./distributed_cvtcolor $(W) $(H)

clean:
	rm -f distributed_cvtcolor
