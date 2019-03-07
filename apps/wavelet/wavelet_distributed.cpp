#include "Halide.h"
#include "mpi_timing.h"

using namespace Halide;

Var x, y, c;

Func haar_x(Func in) {
    Func out;
    out(x, y, c) = select(c == 0,
                          (in(2*x, y) + in(2*x+1, y)),
                          (in(2*x, y) - in(2*x+1, y)))/2;
    out.unroll(c, 2);
    return out;
}

Func inverse_haar_x(Func in) {
    Func out;
    out(x, y) = select(x%2 == 0,
                       in(x/2, y, 0) + in(x/2, y, 1),
                       in(x/2, y, 0) - in(x/2, y, 1));
    out.unroll(x, 2);
    return out;
}


const float D0 = 0.4829629131445341f;
const float D1 = 0.83651630373780772f;
const float D2 = 0.22414386804201339f;
const float D3 = -0.12940952255126034f;

/*
const float D0 = 0.34150635f;
const float D1 = 0.59150635f;
const float D2 = 0.15849365f;
const float D3 = -0.1830127f;
*/

Func daubechies_x(Func in) {
    Func out;
    out(x, y, c) = select(c == 0,
                          D0*in(2*x-1, y) + D1*in(2*x, y) + D2*in(2*x+1, y) + D3*in(2*x+2, y),
                          D3*in(2*x-1, y) - D2*in(2*x, y) + D1*in(2*x+1, y) - D0*in(2*x+2, y));
    out.unroll(c, 2);
    return out;
}

Func inverse_daubechies_x(Func in) {
    Func out;
    out(x, y) = select(x%2 == 0,
                       D2*in(x/2, y, 0) + D1*in(x/2, y, 1) + D0*in(x/2+1, y, 0) + D3*in(x/2+1, y, 1),
                       D3*in(x/2, y, 0) - D0*in(x/2, y, 1) + D1*in(x/2+1, y, 0) - D2*in(x/2+1, y, 1));
    out.unroll(x, 2);
    return out;
}

std::default_random_engine generator(0);
std::uniform_real_distribution<float> distribution(0, 1);

DistributedImage<float> input, output;
Image<float> global_input, global_output;

bool float_eq(float a, float b) {
    const float thresh = 1e-5;
    return a == b || (std::abs(a - b) / b) < thresh;
}

float rndflt() {
    return distribution(generator);
}

Func build(bool distributed) {
    Func clamped;
    if (distributed) {
        clamped(x, y) = input(clamp(x, 0, input.global_width() - 1),
                              clamp(y, 0, input.global_height() - 1));
    } else {
        clamped(x, y) = global_input(clamp(x, 0, global_input.width() - 1),
                                     clamp(y, 0, global_input.height() - 1));
    }

    Func final = daubechies_x(clamped);
    final.parallel(y);

    if (distributed) {
        final.distribute(y);
    }

    return final;
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
    const int ow = w/2, oh = h, od = 2;

    input = DistributedImage<float>(w, h);
    output = DistributedImage<float>(ow, oh, od);
    // global_input = Image<float>(w, h);
    // global_output = Image<float>(ow, oh, od);

    // Func daubechies_correct = build(false);
    Func daubechies_distributed = build(true);

    output.set_domain(x, y, c);
    output.placement().distribute(y);
    output.allocate();
    input.set_domain(x, y);
    input.placement().distribute(y);
    input.allocate(daubechies_distributed, output);

    // for (int y = 0; y < h; y++) {
    //     for (int x = 0; x < w; x++) {
    //         float v = x+y;//rndflt();
    //         if (input.mine(x, y)) {
    //             int lx = input.local(0, x), ly = input.local(1, y);
    //             input(lx, ly) = v;
    //         }
    //         // global_input(x, y) = v;
    //     }
    // }

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = input.global(0, x) + input.global(1, y);
        }
    }


    // JIT compile the pipeline eagerly, so we don't interfere with timing
    Target target = get_target_from_environment();
    daubechies_distributed.compile_jit(target);
    // daubechies_correct.realize(global_output);

    const int niters = 50;
#ifdef USE_MPIP
    MPI_Pcontrol(1);
#endif
    std::vector<std::chrono::duration<double,std::milli>> duration_vector_1;
    for (int i = 0; i < niters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start1 = std::chrono::high_resolution_clock::now();
        daubechies_distributed.realize(output.get_buffer());
	MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }


    // for (int c = 0; c < output.channels(); c++) {
    //     for (int y = 0; y < output.height(); y++) {
    //         for (int x = 0; x < output.width(); x++) {
    //             int gx = output.global(0, x), gy = output.global(1, y), gc = output.global(2, c);
    //             if (!float_eq(output(x, y, c), global_output(gx, gy, gc))) {
    //                 printf("[rank %d] output(%d,%d,%d) = %f instead of %f\n", rank, x, y, c, output(x, y, c), global_output(gx, gy, gc));
    //                 MPI_Abort(MPI_COMM_WORLD, 1);
    //                 MPI_Finalize();
    //                 return -1;
    //             }
    //         }
    //     }
    // }

    if (rank == 0) {
        printf("Wavelet test succeeded!\n");
	print_time("performance_CPU.csv", "wavelet", {"DistHalde"},
		 {median(duration_vector_1)});
    }

    MPI_Finalize();
    return 0;
}
