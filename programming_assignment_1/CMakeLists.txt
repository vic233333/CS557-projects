cmake_minimum_required(VERSION 3.20)
project(LaplacianStencil_0_10 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lap main.cpp Laplacian.cpp)

target_compile_options(lap PRIVATE -O3 -march=native -fopenmp)
target_link_options(lap PRIVATE -fopenmp)
