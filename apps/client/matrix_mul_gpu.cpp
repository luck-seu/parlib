#include <gflags/gflags.h>

#include <iostream>

#include "apps/matrix_mul/matrix_multiplication_cpu.h"
#include "apps/matrix_mul/matrix_multiplication_gpu.h"
#include "apps/matrix_mul/rand_matrix_generator.h"

DEFINE_uint32(n_rows, 3, "Row number of the matrix");
DEFINE_uint32(n_cols, 3, "Column number of the matrix");
DEFINE_uint32(ub, 100, "Upper bound of the random number");
DEFINE_uint32(lb, 0, "Lower bound of the random number");
DEFINE_uint32(rand_seed, 0, "Random seed for matrix generation");
DEFINE_uint32(parallelism, 1, "Number of threads for matrix multiplication");
DEFINE_uint32(n_workers, 1, "Number of workers");
DEFINE_bool(use_gpu, false, "Use GPU for matrix multiplication");

using RandMatrixGenerator = luck::parlib::apps::matrix_mul::RandMatrixGenerator;
using MatrixMultiplicationCPU =
    luck::parlib::apps::matrix_mul::MatrixMultiplicationCPU;
using MatrixMultiplicationGPU =
    luck::parlib::apps::matrix_mul::MatrixMultiplicationGPU;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  RandMatrixGenerator mg(FLAGS_n_rows, FLAGS_n_cols);
  mg.GenerateRandomMatrix(FLAGS_lb, FLAGS_ub, FLAGS_rand_seed);
  uint32_t* lhs = mg.GetMatrix();
  uint32_t* rhs = mg.GetMatrixTranspose();

  if (!FLAGS_use_gpu) {
    MatrixMultiplicationCPU mm(FLAGS_parallelism, FLAGS_n_workers);
    mm.Mul(lhs, rhs, FLAGS_n_rows, FLAGS_n_cols, FLAGS_n_cols, FLAGS_n_rows);
  } else {
    MatrixMultiplicationGPU mm;
    mm.Mul(lhs, rhs, FLAGS_n_rows, FLAGS_n_cols, FLAGS_n_cols, FLAGS_n_rows);
  }

  gflags::ShutDownCommandLineFlags();
  return 0;
}
