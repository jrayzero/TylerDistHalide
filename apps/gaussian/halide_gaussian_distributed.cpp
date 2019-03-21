#include "Halide.h"
#include <cmath>
#include "mpi_timing.h"
#include <iostream>
#include <fstream>
using namespace Halide;

int main(int argc, char* argv[]) {
  int rank = 0, numprocs = 0;
  int req = MPI_THREAD_MULTIPLE, prov;
  MPI_Init_thread(&argc, &argv, req, &prov);
  assert(prov == req);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  float kernelX[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float kernelY[] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

  // give defaults if no inputs
  const int w = argc > 1 ? std::stoi(argv[1]) : 1000;
  const int h = argc > 2 ? std::stoi(argv[2]) : 1000;
  
  DistributedImage<float> input(w, h, 3), output(w, h, 3);

  Func gaussian("gaussian"), gaussian_x("gaussian_x");
  Var x("x"), y("y"), c("c");

  Expr e = 0.0f;
  for (int i = 0; i < 5; ++i) {
    e += input(clamp(x + i, 0, input.global_width()-1), y, c) * kernelX[i];
  }
  gaussian_x(x, y, c) = e;

  Expr f = 0.0f;
  for (int j = 0; j < 5; ++j) {
    f += gaussian_x(x, clamp(y + j, 0, input.global_height()-1), c) * kernelY[j];
  }
  gaussian(x, y, c) = f;//cast<uint8_t>(f);

  Var x_inner, y_inner, x_outer, y_outer, tile_index;
  //    gaussian.tile(x, y, x_outer, y_outer, x_inner, y_inner, 8, 8)
  //            .fuse(x_outer, y_outer, tile_index)
  //            .compute_root();
  //            .parallel(x_outer);
  gaussian.parallel(c);
  gaussian.distribute(y);
  output.set_domain(x,y,c);
  output.placement().distribute(y);
  output.allocate();
  input.set_domain(x,y,c);
  input.placement().distribute(y);
  input.allocate(gaussian, output);

  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      for (int c = 0; c < input.channels(); c++) {
	input(x, y, c) = (x+y+c)*rank;
      }
    }  
  }
    
  gaussian.realize(output);

#ifdef DUMP_RESULTS
	std::string fname = "rank_" + std::to_string(rank) + "_w" + std::to_string(w) + "_h" + std::to_string(h) + ".txt";
	std::ofstream out_file;
	out_file.open(fname);
	for (int i = 0; i < output.height(); i++) {
	  for (int j = 0; j < output.width(); j++) {
	    for (int c = 0; c < output.channels(); c++) {
	      out_file << floor(output(j, i, c)) << " "; 
	    }
	  }
	}
	out_file.close();
#endif


  const int niters = 50;

  std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
  for (int i = 0; i < niters; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start1 = std::chrono::high_resolution_clock::now();
    gaussian.realize(output.get_buffer());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration1 = end1 - start1;
    duration_vector_1.push_back(duration1);
  }

    if (rank == 0) {
        printf("Gaussian test succeeded!\n");
	print_time("performance_CPU.csv", "### gaussian_" + std::to_string(w) + "_" + std::to_string(h), {"DistHalide"},
		 {median(duration_vector_1)});
    }
    MPI_Finalize();

  return 0;
}

