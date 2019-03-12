include ../support/Makefile.inc

W ?=
H ?=

all: distributed_sobel

distributed_sobel: sobel_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) -O3 sobel_distributed.cpp $(LIB_HALIDE) -o distributed_sobel -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS)

run_distributed_sobel: distributed_sobel
	$(MPI_RUN) ./distributed_sobel $(W) $(H)

clean:
	rm -f distributed_sobel
