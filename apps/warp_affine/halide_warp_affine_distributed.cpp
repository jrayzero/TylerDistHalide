#include "Halide.h"
#include "mpi_timing.h"
#include <iostream>
#include <fstream>

using namespace Halide;

Expr mixf(Expr x, Expr y, Expr a) {
    return x * (1.0f-a) + y * a;
}

int main(int argc, char* argv[]) {
    int rank = 0, numprocs = 0;
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    // give defaults if no inputs
    const int w = argc > 1 ? std::stoi(argv[1]) : 1000;
    const int h = argc > 2 ? std::stoi(argv[2]) : 1000;

    DistributedImage<uint8_t> input(w, h);
    DistributedImage<float> output(w, h);

    float a00 = 0.1;
    float a01 = 0.1;
    float a10 = 0.1;
    float a11 = 0.1;
    float b00 = 0.1;
    float b10 = 0.1;

    Func affine{"affine"};
    Var x, y;

    // Translating this algorithm as close as possible
    Expr o_r = a11 * y + a10 * x + b00;
    Expr o_c = a01 * y + a00 * x + b10;

    Expr r = o_r - floor(o_r);
    Expr c = o_c - floor(o_c);

    Expr coord_00_r = cast<int>(floor(o_r));
    Expr coord_00_c = cast<int>(floor(o_c));
    Expr coord_01_r = coord_00_r;
    Expr coord_01_c = coord_00_c + 1;
    Expr coord_10_r = coord_00_r + 1;
    Expr coord_10_c = coord_00_c;
    Expr coord_11_r = coord_00_r + 1;
    Expr coord_11_c = coord_00_c + 1;

    coord_00_r = clamp(coord_00_r, 0, input.global_height()-1);
    coord_00_c = clamp(coord_00_c, 0, input.global_width()-1);
    coord_01_r = clamp(coord_01_r, 0, input.global_height()-1);
    coord_01_c = clamp(coord_01_c, 0, input.global_width()-1);
    coord_10_r = clamp(coord_10_r, 0, input.global_height()-1);
    coord_10_c = clamp(coord_10_c, 0, input.global_width()-1);
    coord_11_r = clamp(coord_11_r, 0, input.global_height()-1);
    coord_11_c = clamp(coord_11_c, 0, input.global_width()-1);
    
    Expr A00 = input(coord_00_r, coord_00_c);
    Expr A10 = input(coord_10_r, coord_10_c);
    Expr A01 = input(coord_01_r, coord_01_c);
    Expr A11 = input(coord_11_r, coord_11_c);

    affine(x, y) = mixf(mixf(A00, A10, r), mixf(A01, A11, r), c);       
    
    //affine.parallel(y).vectorize(x, 16, Halide::TailStrategy::GuardWithIf);

    affine.distribute(y);
    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();
    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate(affine, output);
    
    for (int y = 0; y < input.height(); y++) {
      for (int x = 0; x < input.width(); x++) {
	input(x, y) = (x+y+rank);
      }
    }

    affine.realize(output);
#ifdef DUMP_RESULTS
    std::string fname = "rank_" + std::to_string(rank) + "_w" + std::to_string(w) + "_h" + std::to_string(h) + ".txt";
    std::ofstream out_file;
    out_file.open(fname);
    for (int i = 0; i < output.height(); i++) {
      for (int j = 0; j < output.width(); j++) {
	out_file << output(j, i) << " "; 
      }
    }
    out_file.close();
#endif

    const int niters = 50;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    for (int i = 0; i < niters; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      auto start1 = std::chrono::high_resolution_clock::now();
      affine.realize(output.get_buffer());
      MPI_Barrier(MPI_COMM_WORLD);
      auto end1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double,std::milli> duration1 = end1 - start1;
      duration_vector_1.push_back(duration1);
    }

    if (rank == 0) {
      printf("Warp affine test succeeded!\n");
      print_time("performance_CPU.csv", "### warp_affine_" + std::to_string(w) + "_" + std::to_string(h), {"DistHalide"},
		 {median(duration_vector_1)});
    }
    MPI_Finalize();

    return 0;
}
