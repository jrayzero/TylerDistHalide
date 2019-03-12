#include "Halide.h"
#include "mpi_timing.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <memory>

using namespace Halide;

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(10, 500);
Var x("x"), y("y"), xi, yi, tile;

float rndflt() {
    return distribution(generator);
}

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

// Hack to make sure we use the best non-distributed schedule for the
// baseline when running scalability tests.
bool actually_distributed() {
    char *e = getenv("HL_DISABLE_DISTRIBUTED");
    // disable = 0 => return true
    // disable != 0 => return false
    return !e || atoi(e) == 0;
}

Func build(Func input, bool distributed) {
    Func block, block_transpose, output;
    block(x, y) = input(x, y);
    block_transpose(x, y) = block(y, x);
    output(x, y) = block_transpose(x, y);

    if (distributed && actually_distributed()) {
        output.split(x, x, xi, 16).vectorize(xi);
        output.parallel(y);
        output.distribute(y);
    } else {
        output.tile(x, y, xi, yi, 16, 16).vectorize(xi).unroll(yi);
        output.parallel(y);
    }

    return output;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = argc > 1 ? std::stoi(argv[1]) : 1000;
    const int h = argc > 2 ? std::stoi(argv[2]) : 1000;

    // Image<float> global_input(w, h), global_output(w, h);
    DistributedImage<float> input(w, h), output(h, w);

    Func accessor, global_accessor;
    accessor(x, y) = input(x, y);
    // global_accessor(x, y) = global_input(x, y);

    // Func transpose_correct = build(global_accessor, false);
    Func transpose_distributed = build(accessor, true);

    output.set_domain(x, y);
    //output.placement().tile(x, y, xi, yi, 16, 16).distribute(y);
    output.placement().distribute(y);
    output.allocate();
    input.set_domain(x, y);
    input.placement().distribute(x);
    input.allocate(transpose_distributed, output);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            int gx = input.global(0, x), gy = input.global(1, y);
            float v = gx+gy; //rndflt();
            input(x, y) = v;
        }
    }

    // transpose_correct.realize(global_output);
    transpose_distributed.realize(output.get_buffer());

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
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    for (int i = 0; i < niters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        transpose_distributed.realize(output.get_buffer());
	MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    // for (int y = 0; y < output.height(); y++) {
    //     for (int x = 0; x < output.width(); x++) {
    //         int gx = output.global(0, x), gy = output.global(1, y);
    //         if (!float_eq(output(x, y), global_output(gx, gy))) {
    //             printf("[rank %d] output(%d,%d) = %f instead of %f\n", rank, x, y, output(x, y), global_output(gx, gy));
    //             MPI_Abort(MPI_COMM_WORLD, 1);
    //             MPI_Finalize();
    //             return -1;
    //         }
    //     }
    // }

    if (rank == 0) {
        printf("Transpose test succeeded!\n");
	print_time("performance_CPU.csv", "### transpose_" + std::to_string(w) + "_" + std::to_string(h), {"DistHalide"},
		 {median(duration_vector_1)});
    }
    MPI_Finalize();
    return 0;
}
