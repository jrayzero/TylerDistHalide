#include "Halide.h"
#include <iostream>
#include <fstream>
using namespace Halide;

#include "halide_image_io.h"
#include "mpi_timing.h"

DistributedImage<float> input, output;

Var x("x"), y("y");

Func build(bool distributed) {
    Func clamped;
    if (distributed) {
        clamped(x, y) = input(clamp(x, 0, input.global_width() - 1),
                              clamp(y, 0, input.global_height() - 1));
    } else {
        clamped(x, y) = global_input(clamp(x, 0, global_input.width() - 1),
                                     clamp(y, 0, global_input.height() - 1));
    }
    Func sobelx, sobely;
    sobelx(x, y) = -1*clamped(x-1, y-1) + clamped(x+1, y-1) +
        -2*clamped(x-1, y) + 2*clamped(x+1, y) +
        -1*clamped(x-1, y+1) + clamped(x+1, y+1);

    sobely(x, y) = -1*clamped(x-1, y-1) + -2*clamped(x, y-1) + -1*clamped(x+1,y-1) +
        clamped(x-1, y+1) + 2*clamped(x, y+1) + clamped(x+1, y+1);

    Func sobel;
    sobel(x, y) = sqrt(sobelx(x, y) * sobelx(x, y) + sobely(x, y) * sobely(x, y));

    sobelx.vectorize(x, 8).compute_at(sobel, y);
    sobely.vectorize(x, 8).compute_at(sobel, y);

    if (distributed) {
        sobel.distribute(y);
    }
    return sobel;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    std::cerr << "Total ranks: " << numprocs << std::endl;
    const int w = argc > 1 ? std::stoi(argv[1]) : 1000;
    const int h = argc > 2 ? std::stoi(argv[2]) : 1000;

    input = DistributedImage<float>(w, h);
    output = DistributedImage<float>(w, h);

    Func sobel_distributed = build(true);

    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();

    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate(sobel_distributed, output);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
	  input(x, y) = (rank+x+y);
        }
    }
    sobel_distributed.compile_to_lowered_stmt("blah.txt", {input});
    sobel_distributed.realize(output.get_buffer());

    /*#ifdef DUMP_RESULTS
    std::string fname = "rank_" + std::to_string(rank) + "_w" + std::to_string(w) + "_h" + std::to_string(h) + ".txt";
    std::ofstream out_file;
    out_file.open(fname);
    for (int i = 0; i < output.height(); i++) {
      for (int j = 0; j < output.width(); j++) {
	out_file << output(j, i) << " "; 
      }
    }
    out_file.close();
    #endif*/

    const int niters = 50;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    for (int i = 0; i < niters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        sobel_distributed.realize(output.get_buffer());
	MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    if (rank == 0) {
        printf("Sobel test succeeded!\n");
	print_time("performance_CPU.csv", "### sobel_" + std::to_string(w) + "_" + std::to_string(h), {"DistHalide"},
		 {median(duration_vector_1)});
    }

    MPI_Finalize();
    return 0;
}
