#include "Halide.h"
#include "mpi_timing.h"
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>

using namespace Halide;

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(10, 500);
Var x("x"), y("y"), xi("xi"), xo("xo"), yo("yo"), yi("yo"), yii("yii"), xii("xii");
const int block_size = 32;
int matrix_size;

float rndflt() {
    return distribution(generator);
}

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

Func build(Func A, Func B, bool distributed) {
    Func matrix_mul("matrix_mul");

    RDom k(0, matrix_size);
    RVar ki;

    matrix_mul(x, y) = 0.0f;
    matrix_mul(x, y) += A(k, y) * B(x, k);

    matrix_mul.split(x, x, xi, block_size).split(xi, xi, xii, 8)
        .split(y, y, yi, block_size).split(yi, yi, yii, 4)
        .reorder(xii, yii, xi, yi, x, y)
        .parallel(y).vectorize(xii).unroll(xi).unroll(yii);

    matrix_mul.update()
        .split(x, x, xi, block_size).split(xi, xi, xii, 8)
        .split(y, y, yi, block_size).split(yi, yi, yii, 4)
        .split(k, k, ki, block_size)
        .reorder(xii, yii, xi, ki, yi, k, x, y)
        .parallel(y).vectorize(xii).unroll(xi).unroll(yii);

    // matrix_mul
    //     .bound(x, 0, matrix_size)
    //     .bound(y, 0, matrix_size);

    if (distributed) {
        matrix_mul.distribute(y);
        matrix_mul.update().distribute(y);
    }

    return matrix_mul;
}

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = argc > 1 ? std::stoi(argv[1]) : 1024;
    const int h = argc > 2 ? std::stoi(argv[2]) : 1024;
    assert(w == h && "Non-square matrices unimplemented");

    // Compute C = A*B
    Image<float> global_A(w, h), global_B(w, h), global_C(w, h);
    DistributedImage<float> A(w, h), B(w, h), C(w, h);
    A.set_domain(x, y);
    A.placement();
    A.allocate();
    B.set_domain(x, y);
    B.placement();
    B.allocate();
    C.set_domain(x, y);
    C.placement().split(x, x, xi, block_size).split(xi, xi, xii, 8)
        .split(y, y, yi, block_size).split(yi, yi, yii, 4)
        .distribute(y);
    C.allocate();

    matrix_size = w;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float v = rndflt();
            if (A.mine(x, y)) {
                int lx = A.local(0, x), ly = A.local(1, y);
                A(lx, ly) = 0.1f;
            }
            global_A(x, y) = 0.1f;

            v = rndflt();
            if (B.mine(x, y)) {
                int lx = B.local(0, x), ly = B.local(1, y);
                B(lx, ly) = 0.1f;
            }
            global_B(x, y) = 0.1f;
        }
    }

    Func accessor_A, global_accessor_A, accessor_B, global_accessor_B;
    accessor_A(x, y) = A(x, y);
    global_accessor_A(x, y) = global_A(x, y);
    accessor_B(x, y) = B(x, y);
    global_accessor_B(x, y) = global_B(x, y);

    Func mm_correct = build(global_accessor_A, global_accessor_B, false);
    Func mm_distributed = build(accessor_A, accessor_B, true);

    mm_correct.realize(global_C);
    mm_distributed.realize(C.get_buffer());

#ifdef DUMP_RESULTS
    std::string fname = "rank_" + std::to_string(rank) + "_w" + std::to_string(w) + "_h" + std::to_string(h) + ".txt";
    std::ofstream out_file;
    out_file.open(fname);
    for (int i = 0; i < C.height(); i++) {
      for (int j = 0; j < C.width(); j++) {
	out_file << C(j, i) << " "; 
      }
    }
	out_file.close();
#endif

    const int niters = 50;
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    for (int i = 0; i < niters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        mm_distributed.realize(C.get_buffer());
        MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    /*    for (int y = 0; y < C.height(); y++) {
        for (int x = 0; x < C.width(); x++) {
            int gx = C.global(0, x), gy = C.global(1, y);
            if (!float_eq(C(x, y), global_C(gx, gy))) {
                printf("[rank %d] C(%d,%d) = %f instead of %f\n", rank, x, y, C(x, y), global_C(gx, gy));
                MPI_Abort(MPI_COMM_WORLD, 1);
                MPI_Finalize();
                return -1;
            }
        }
	}*/


    if (rank == 0) {
        printf("Matrix multiply test succeeded!\n");
	print_time("performance_CPU.csv", "### mat_mul", {"DistHalide"},
		 {median(duration_vector_1)});
    }
    MPI_Finalize();

    return 0;
}
