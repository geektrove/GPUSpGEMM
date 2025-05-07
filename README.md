# SpGEMM on GPU

This repository contains the implementation of the master's thesis "Sparse general matrix-matrix multiplication on GPU". It focuses on a novel binary search accumulator for SpGEMM on NVIDIA GPUs. The main implementation is called 'proposal' and is compared against cuSPARSE and OpSparse implementations to demonstrate its effectiveness.

## Project Structure

- `src/apps/`: Executable applications for benchmarking and validation
- `src/libproposal/`: Main library implementing the proposed SpGEMM algorithm
- `src/libopsparse/`: Library implementing the [OpSparse](https://github.com/lorentzbf/OpSparse) algorithm, modified to be used as a CMake subdirectory with the custom .csr format
- `src/libcusparse/`: Library implementing the cuSPARSE 1 and cuSPARSE 2 algorithms
- `src/libutils/`: Utility functions for CUDA, sparse matrices, and benchmarking

## Dependencies

External dependencies have to be provided by the user.

- CMake 3.24 or later (might work with older versions, but not tested)
- Ninja 1.12 or later (might work with older versions, but not tested)
- CUDA 12.8 or later

Other dependencies are fetched using CMake's `FetchContent` module, see `cmake/dependencies.cmake` for details.

- gsl-lite 0.42.0
- fmt 11.1.4
- spdlog 1.15.1
- nvtx3 3.1.1
- CLI11 2.5.0

In order to be able to use `tasks.py` file ([invoke](https://www.pyinvoke.org/) task runner) to download the matrices, run benchmarks locally, or on the clusters, or analyze the results in the jupyter notebook `notebooks/analyze-results.ipynb`, you need to install python dependencies.

```bash
pip install -r requirements.txt
```

## Compute capabilities

The code includes parameters for the following list of compute capabilities:

- 8.0
- 8.6
- 8.9

Please change the `CMAKE_CUDA_ARCHITECTURES` variable in the `CMakePresets.json` file to add or remove compute capabilities. You can also use the [-DCMAKE_CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html) flag or [CUDAARCHS](https://cmake.org/cmake/help/latest/envvar/CUDAARCHS.html) environment variable to specify the compute capabilities. The values such as `all`, `all-major` and `native` are not supported.

If you want to build for another architecture, please update the `src/libproposal/include/proposal/parameters.cuh` file with appropriate parameters for your target compute capability.

## Building and running the release version

### Build the proposed implementation with double precision

```bash
cmake --preset=release -DUSE_DOUBLE_PRECISION=ON
cmake --build --preset=release --target=proposal
```

### Download matrices and convert them to the custom .csr binary format

```bash
invoke download-matrices
```

### Run benchmarks

```bash
invoke benchmark --target=proposal
```

## ClusterFIT and Cluster Star

The `tasks.py` file also contains two commands to run the benchmarks on the clusters. For example, to submit the benchmark on ClusterFIT, compile all targets in release mode and run `invoke run-fitcluster`. This will submit SLURM jobs for each target and save the results in the `results/fitcluster` directory. The author's results are provided as a part of this repository.

In order to analyze the results, see the `notebooks/analyze-results.ipynb` file. The figures in the notebook are saved in the `figures` directory and are also included.

## Development and testing

The `ccache` is used to cache the build results, so consider installing it or creating your own CMake preset that does not use it.

### Build the debug version with warnings

```bash
cmake --preset=dev
cmake --build --preset=dev --target=proposal
```

### Run with logging

```bash
./build/dev/src/apps/proposal data/webbase-1M.csr data/webbase-1M.csr --loglevel=debug benchmark
```

### Run with compute sanitizer

```bash
compute-sanitizer --tool=memcheck ./build/dev/src/apps/proposal data/webbase-1M.csr data/webbase-1M.csr benchmark
```

### Validate the result against cuSPARSE 2 algorithm

```bash
./build/dev/src/apps/proposal data/webbase-1M.csr data/webbase-1M.csr validate
```

### Save the output matrix to a file for later analysis

```bash
./build/dev/src/apps/proposal data/webbase-1M.csr data/webbase-1M.csr save /tmp/webbase-1M-squared.csr
```
