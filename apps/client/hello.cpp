#include <gflags/gflags.h>

#include <iostream>

#include "apps/matrix_generator.h"
#include "apps/matrix_multiplication_cpu.h"

DEFINE_uint64(n_rows, 3, "Row number of the matrix");
DEFINE_uint64(n_cols, 3, "Column number of the matrix");
DEFINE_uint64(ub, 100, "Upper bound of the random number");
DEFINE_uint64(lb, 0, "Lower bound of the random number");
DEFINE_uint64(rand_seed, 0, "Random seed for matrix generation");
DEFINE_uint64(parallelism, 1, "Number of threads for matrix multiplication");
DEFINE_uint64(n_workers, 1, "Number of workers");

using MatrixGenerator = luck::parlib::apps::MatrixGenerator;
using MatrixMultiplicationCPU =
    luck::parlib::apps::MatrixMultiplicationCPU;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  MatrixGenerator mg(FLAGS_n_rows, FLAGS_n_cols);
  mg.GenerateRandomMatrix(FLAGS_lb, FLAGS_ub, FLAGS_rand_seed);

  MatrixMultiplicationCPU mm(FLAGS_parallelism, FLAGS_n_workers);

  mm.Mul(mg.GetMatrix(), mg.GetMatrix(), FLAGS_n_rows, FLAGS_n_cols);

  gflags::ShutDownCommandLineFlags();
  return 0;
}
