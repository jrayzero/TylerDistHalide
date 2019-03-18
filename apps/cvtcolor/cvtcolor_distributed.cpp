#include "Halide.h"
#include <iostream>
#include <fstream>
using namespace Halide;

#include "halide_image_io.h"
#include "mpi_timing.h"

DistributedImage<uint32_t> input, output;

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

int main(int argc, char **argv) {
    int req = MPI_THREAD_MULTIPLE, prov;
    MPI_Init_thread(&argc, &argv, req, &prov);
    assert(prov == req);
    int rank = 0, numprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int w = argc > 1 ? std::stoi(argv[1]) : 1000;
    const int h = argc > 2 ? std::stoi(argv[2]) : 1000;
    const int d = 3;

    input = DistributedImage<uint32_t>(w, h, d);
    output = DistributedImage<uint32_t>(w, h);

    Func RGB2Gray{"RGB2Gray"};
    Var x("x"), y("y"), c("c");

    const Expr yuv_shift = cast<uint32_t>(14);
    const Expr R2Y = cast<uint32_t>(4899);
    const Expr G2Y = cast<uint32_t>(9617);
    const Expr B2Y = cast<uint32_t>(1868);

    RGB2Gray(x, y) = cast<uint32_t>(CV_DESCALE( (input(x, y, 2) * B2Y
                                                + input(x, y, 1) * G2Y
                                                + input(x, y, 0) * R2Y),
                                               yuv_shift));

    //    RGB2Gray.parallel(y).vectorize(x, 8);
    RGB2Gray.distribute(y);

    output.set_domain(x, y);
    output.placement().distribute(y);
    output.allocate();

    input.set_domain(x, y, c);
    input.placement().distribute(y);
    input.allocate();//RGB2Gray, output);
    
    for (int y = 0; y < input.height(); y++) {
      for (int x = 0; x < input.width(); x++) {
	for (int c = 0; c < input.channels(); c++) {
	  input(x, y, c) = (rank+x+y+c);
	}
      }
    }

    RGB2Gray.realize(output.get_buffer());
    
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
        RGB2Gray.realize(output.get_buffer());
	MPI_Barrier(MPI_COMM_WORLD);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> duration1 = end1 - start1;
        duration_vector_1.push_back(duration1);
    }

    if (rank == 0) {
        printf("cvtcolor test succeeded!\n");
	print_time("performance_CPU.csv", "### cvtcolor_" + std::to_string(w) + "_" + std::to_string(h), {"DistHalide"},
		 {median(duration_vector_1)});
    }
    
    MPI_Finalize();
    return 0;
}
