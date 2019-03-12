#include "Halide.h"
#include "mpi_timing.h"
#include <iostream>
#include <fstream>
using namespace Halide;

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    auto proc_grid = approx_factors_near_cubert(numprocs);
    int p = proc_grid[0], q = proc_grid[1], r = proc_grid[2];
    if (rank == 0) printf("Using process grid %dx%dx%d\n", p, q, r);

    const int w = argc > 1 ? std::stoi(argv[1]) : 100;
    const int h = argc > 2 ? std::stoi(argv[2]) : 100;
    const int d = argc > 3 ? std::stoi(argv[3]) : 10;
    Var x("x"), y("y"), z("z"), xi("xi"), yi("yi");

    // Declare our input and output in global width and height.
    DistributedImage<float> input(w, h, d);
    DistributedImage<float> output(w, h, d);

    Func clamped;
    clamped(x, y, z) = input(clamp(x, 0, input.global_width() - 1),
                             clamp(y, 0, input.global_height() - 1),
                             clamp(z, 0, input.global_channels() - 1));
    Func heat3d("heat3d"), buffered;
    //Expr c0 = cast<double>(0.5f), c1 = cast<double>(-0.25f);
    Expr c0 = 0.5f, c1 = -0.25f;
    buffered(x, y, z) = clamped(x, y, z);
    heat3d(x, y, z) = c0 * buffered(x, y, z) + c1 * (buffered(x-1, y, z) + buffered(x+1, y, z) +
                                                  buffered(x, y-1, z) + buffered(x, y+1, z) +
                                                  buffered(x, y, z-1) + buffered(x, y, z+1));
    // This has the effect of double-buffering the input:
    // buffered.compute_root()
    //     .tile(x, y, xi, yi, 8, 8).vectorize(xi).unroll(yi)
    //     .parallel(z)
    //     .distribute(x, y, z, p, q, r);
    heat3d
        .tile(x, y, xi, yi, 8, 8).vectorize(xi).unroll(yi)
        .parallel(z)
        .distribute(x, y, z, p, q, r);

    output.set_domain(x, y, z);
    output.placement().tile(x, y, xi, yi, 8, 8).distribute(x, y, z, p, q, r);
    output.allocate();

    input.set_domain(x, y, z);
    input.placement().tile(x, y, xi, yi, 8, 8).distribute(x, y, z, p, q, r);
    input.allocate(heat3d, output);

    for (int z = 0; z < input.channels(); z++) {
        for (int y = 0; y < input.height(); y++) {
            for (int x = 0; x < input.width(); x++) {
	      input(x, y, z) = (x+y+z);
            }
        }
    }

    heat3d.compile_jit();

    heat3d.realize(output);
#ifdef DUMP_RESULTS
    std::string fname = "rank_" + std::to_string(rank) + "_w" + std::to_string(w) + "_h" + std::to_string(h) + "_d" + std::to_string(d) + ".txt";
    std::ofstream out_file;
    out_file.open(fname);
    for (int i = 0; i < output.height(); i++) {
      for (int j = 0; j < output.width(); j++) {
	for (int k = 0; k < output.channels(); k++) {
	  out_file << output(j, i, k) << " "; 
	}
      }
    }
    out_file.close();
#endif
    
    // Run the program and test output for correctness
    const int niters = 50;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    for (int i = 0; i < niters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        heat3d.realize(output);
        MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }
    if (rank == 0) {
        printf("Heat test succeeded!\n");
	print_time("performance_CPU.csv", "### heat", {"DistHalide"},
		 {median(duration_vector_1)});
    }
    MPI_Finalize();
    return 0;
}
