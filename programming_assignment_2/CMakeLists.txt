cmake_minimum_required(VERSION 3.20)
project(programming_assignment_2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

set(SRCS
        main.cpp
        ConjugateGradients.cpp
        Laplacian.cpp
        PointwiseOps.cpp
        Reductions.cpp
        Utilities.cpp
        MergedKernels.cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp=libomp")
set(OPENMP_LIB "C:/Program Files/LLVM/lib/libomp.lib")

# Task 1: Baseline
add_executable(lap_baseline ${SRCS})
target_link_libraries(lap_baseline PRIVATE "${OPENMP_LIB}")

# Task 2: Merged kernels
add_executable(lap_merged ${SRCS})
target_link_libraries(lap_merged PRIVATE "${OPENMP_LIB}")
target_compile_definitions(lap_merged PRIVATE USE_MERGED)

# Task 3: Baseline Serial
add_executable(lap_baseline_serial ${SRCS})
target_link_libraries(lap_baseline_serial PRIVATE "${OPENMP_LIB}")
target_compile_definitions(lap_baseline_serial PRIVATE NUM_THREADS=1)

# Task 4: Merged Serial
add_executable(lap_merged_serial ${SRCS})
target_link_libraries(lap_merged_serial PRIVATE "${OPENMP_LIB}")
target_compile_definitions(lap_merged_serial PRIVATE USE_MERGED NUM_THREADS=1)

enable_testing()

# Define environment variables for OpenMP settings
# For 12 threads:
set(OMP_ENV_12_THREADS
        "OMP_DYNAMIC=FALSE"
        "OMP_PROC_BIND=close"
        "OMP_PLACES=cores"
        "OMP_NUM_THREADS=12"
)

# For 1 thread:
set(OMP_ENV_1_THREAD
        "OMP_DYNAMIC=FALSE"
        "OMP_PROC_BIND=close"
        "OMP_PLACES=cores"
        "OMP_NUM_THREADS=1"
)

# 1. Run Baseline (12 threads)
add_test(NAME run_baseline COMMAND lap_baseline)
set_tests_properties(run_baseline PROPERTIES ENVIRONMENT "${OMP_ENV_12_THREADS}")

# 2. Run Merged (12 threads)
add_test(NAME run_merged COMMAND lap_merged)
set_tests_properties(run_merged PROPERTIES ENVIRONMENT "${OMP_ENV_12_THREADS}")

# 3. Run Baseline Serial (1 thread)
add_test(NAME run_baseline_serial COMMAND lap_baseline_serial)
set_tests_properties(run_baseline_serial PROPERTIES ENVIRONMENT "${OMP_ENV_1_THREAD}")

# 4. Run Merged Serial (1 thread)
add_test(NAME run_merged_serial COMMAND lap_merged_serial)
set_tests_properties(run_merged_serial PROPERTIES ENVIRONMENT "${OMP_ENV_1_THREAD}")