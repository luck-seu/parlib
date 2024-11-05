# ParLib: An Uniform Library for Hybrid Parallelism

> ParLib is a CPU-GPU collaborative parallel library built atop CMake, with GCC and NVCC mixed compilation.

## Dependencies
- A modern C++ compiler compliant with the C++17 standard (gcc >= 9)
- CMake (>= 2.8)
- Facebook folly library (>= v2022.11.28.00)
- nvcc (>= 12.4)
- gflags (>= 2.2)

## Build
First, clone the project and install dependencies on your environment.
```bash
# Clone the project (SSH).
# Make sure you have your public key has been uploaded to GitHub!
git clone git@github.com:luck-seu/parlib.git
# Install dependencies.
$SRC_DIR=`parlib` # top-level MiniGraph source dir
$cd $SRC_DIR
$./dependencies.sh
```

Then, build the project.
```bash
./build.sh
```

## Running `MatrixMultiplication` as an example
Matrix multiplication by CPU multicore parallelism:
```bash
./bin/matrix_mul_gpu_exec -n_rows 5000 -n_cols 2000 -lb 0 -ub 100 -parallelism 80 -n_workers 256
```

Matrix multiplication by GPU SIMD parallelism:
```bash
./bin/matrix_mul_gpu_exec -n_rows 5000 -n_cols 2000 -lb 0 -ub 100 -use_gpu
```

where,
- `-n_rows` specifies the #rows of the matrix;
- `-n_cols` specifies the #cols of the matrix (the matrix multiplication takes an matrix and its transpose as input, hecne we only need to specify the shape of a single matrix);
- `-lb` specifies the minimum value that could exists in the matrix;
- `-ub` specifies the maximum value that could exists in the matrix;
- `-parallelism` specifies the maximum parallelism on CPU;
- `-n_workers` specifies the granularity of the task unit (i.e., the task will be partitioned into `n_workers` task units, each of which will be processed by a thread (i.e., worker));
- `-use_gpu` indicates whether the GPU acceleration is enabled. The default parallelism of GPU is 64 * 64;


## Remark
Applications that should be compiled by nvcc should ended with `_gpu.cpp` (e.g., `apps/client/matrix_mul_gpu.cpp`).


## Acknowledgement

This project originates from several parallel systems developed at the Shenzhen Institute of Computer Science:
- [MiniGraph](https://github.com/SICS-Fundamental-Research-Center/MiniGraph)
- [Planar](https://github.com/SICS-Fundamental-Research-Center/MiniGraph)
- [Hyperlocker](https://github.com/SICS-Fundamental-Research-Center/HyperBlocker)

Thanks to their authors: Shuhao Liu (SICS), Xiaoke Zhu (BUAA), Yang Liu (BUAA).