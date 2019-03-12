include ../support/Makefile.inc

W ?=
H ?=

.PHONY: clean

interpolate: interpolate.cpp
	$(CXX) $(CXXFLAGS) interpolate.cpp $(LIB_HALIDE) -o interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

distributed_interpolate: interpolate_distributed.cpp
	$(CXX) $(MPI_CXX_FLAGS) $(CXXFLAGS) interpolate_distributed.cpp $(LIB_HALIDE) -o distributed_interpolate -lpthread -ldl -lz \
	$(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS) $(LDFLAGS) $(LLVM_SHARED_LIBS)

run_distributed_interpolate: distributed_interpolate
	$(MPI_RUN) ./distributed_interpolate $(W) $(H)

out.png: interpolate
	./interpolate ../images/rgba.png out.png

clean:
	rm -f interpolate interpolate.h out.png distributed_interpolate
