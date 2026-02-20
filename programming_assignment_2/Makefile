# Use Microsoft Visual Studio Build Tools and LLVM toolchain to compile the project (Windows)
VSDEV := "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"

# Paths to LLVM binaries and libraries
LLVM_BIN := C:\Program Files\LLVM\bin
LLVM_LIB := C:\Program Files\LLVM\lib

CXX := "$(LLVM_BIN)\clang++.exe"

CXXFLAGS := -O3 -march=native -std=c++23 -fopenmp=libomp
LIBOMP := "$(LLVM_LIB)\libomp.lib"

SRCS := main.cpp ConjugateGradients.cpp Laplacian.cpp \
        PointwiseOps.cpp Reductions.cpp Utilities.cpp \
        MergedKernels.cpp

# Task 1: baseline with per-kernel timing
build_base:
	call $(VSDEV) -arch=x64 -host_arch=x64 && \
	$(CXX) $(SRCS) -o lap_baseline.exe $(CXXFLAGS) $(LIBOMP)

# Task 2: merged kernels
build_merged:
	call $(VSDEV) -arch=x64 -host_arch=x64 && \
	$(CXX) $(SRCS) -o lap_merged.exe $(CXXFLAGS) $(LIBOMP) -DUSE_MERGED

# Single-thread versions (for report comparison)
build_serial:
	call $(VSDEV) -arch=x64 -host_arch=x64 && \
	$(CXX) $(SRCS) -o lap_baseline_serial.exe $(CXXFLAGS) $(LIBOMP) -DNUM_THREADS=1

build_merged_serial:
	call $(VSDEV) -arch=x64 -host_arch=x64 && \
	$(CXX) $(SRCS) -o lap_merged_serial.exe $(CXXFLAGS) $(LIBOMP) -DUSE_MERGED -DNUM_THREADS=1

build_all: build_base build_merged build_serial build_merged_serial

run: build_all
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=12 && .\lap_baseline.exe
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=12 && .\lap_merged.exe
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=1 && .\lap_baseline_serial.exe
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=1 && .\lap_merged_serial.exe

clean:
	del /Q *.exe