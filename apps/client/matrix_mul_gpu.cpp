#include <gflags/gflags.h>

#include <iostream>

#include "apps/flat_matrix_generator.h"
#include "apps/matrix_multiplication_gpu.h"

DEFINE_uint64(n_rows, 3, "Row number of the matrix");
DEFINE_uint64(n_cols, 3, "Column number of the matrix");
DEFINE_uint64(ub, 100, "Upper bound of the random number");
DEFINE_uint64(lb, 0, "Lower bound of the random number");
DEFINE_uint64(rand_seed, 0, "Random seed for matrix generation");

using FlatMatrixGenerator = luck::parlib::apps::FlatMatrixGenerator;
using MatrixMultiplicationGPU = luck::parlib::apps::MatrixMultiplicationGPU;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FlatMatrixGenerator mg(FLAGS_n_rows, FLAGS_n_cols);
  mg.GenerateRandomMatrix(FLAGS_lb, FLAGS_ub, FLAGS_rand_seed);

  MatrixMultiplicationGPU mm;

  mm.Mul(mg.GetMatrix(), mg.GetMatrix(), FLAGS_n_rows, FLAGS_n_cols);

  gflags::ShutDownCommandLineFlags();
  return 0;
}
