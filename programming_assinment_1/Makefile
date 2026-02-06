# Use Microsoft Visual Studio Build Tools and LLVM toolchain to compile the project (Windows)
VSDEV := "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"

# Paths to LLVM binaries and libraries
LLVM_BIN := "C:\Program Files\LLVM\bin"
LLVM_LIB := "C:\Program Files\LLVM\lib"

CXX := $(LLVM_BIN)\clang++.exe
EXE := lap.exe
KJI_EXE := lap_kji.exe
IKJ_EXE := lap_ikj.exe

CXXFLAGS := -O3 -march=native -std=c++23 -fopenmp=libomp
LIBOMP := $(LLVM_LIB)\libomp.lib
LIBOMPDLL := $(LLVM_BIN)\libomp.dll

build-windows:
	call $(VSDEV) -arch=x64 -host_arch=x64 && $(CXX) main.cpp Laplacian.cpp -o $(EXE) $(CXXFLAGS) $(LIBOMP)
	call $(VSDEV) -arch=x64 -host_arch=x64 && $(CXX) main.cpp Laplacian_KJI.cpp -o $(KJI_EXE) $(CXXFLAGS) $(LIBOMP)
	call $(VSDEV) -arch=x64 -host_arch=x64 && $(CXX) main.cpp Laplacian_IKJ.cpp -o $(IKJ_EXE) $(CXXFLAGS) $(LIBOMP)

# Use g++ to compile the project (Linux)
build-linux:
	g++ main.cpp Laplacian.cpp -o $(EXE) -O3 -march=native -std=c++23 -fopenmp
	g++ main.cpp Laplacian_KJI.cpp -o $(KJI_EXE) -O3 -march=native -std=c++23 -fopenmp
	g++ main.cpp Laplacian_IKJ.cpp -o $(IKJ_EXE) -O3 -march=native -std=c++23 -fopenmp

run:
	.\$(EXE)

clean-windows:
	-del /Q $(EXE) 2>NUL
	-del /Q $(KJI_EXE) 2>NUL
	-del /Q $(IKJ_EXE) 2>NUL

clean-linux:
	rm -f $(EXE)
	rm -f $(KJI_EXE)
	rm -f $(IKJ_EXE)

test:
	set OMP_NUM_THREADS=1&& .\$(EXE)
	set OMP_NUM_THREADS=2&& .\$(EXE)
	set OMP_NUM_THREADS=4&& .\$(EXE)
	set OMP_NUM_THREADS=8&& .\$(EXE)
	set OMP_NUM_THREADS=12&& .\$(EXE)
	set OMP_NUM_THREADS=16&& .\$(EXE)
	set OMP_NUM_THREADS=24&& .\$(EXE)

test-dynamic-close:
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=1&&  .\$(EXE)
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=2&&  .\$(EXE)
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=4&&  .\$(EXE)
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=8&&  .\$(EXE)
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=12&& .\$(EXE)
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=16&& .\$(EXE)
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=24&& .\$(EXE)

exp-windows:
	echo ===== BASE (i-j-k) with dynamic=false, proc_bind=close, places=cores, threads=12 =====
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=12&& .\$(EXE)
	echo ===== KJI (k-j-i) with dynamic=false, proc_bind=close, places=cores, threads=12 =====
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=12&& .\$(KJI_EXE)
	echo ===== IKJ (i-k-j) with dynamic=false, proc_bind=close, places=cores, threads=12 =====
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=12&& .\$(IKJ_EXE)
	echo ===== BASE (i-j-k) with dynamic=false, proc_bind=close, places=cores, threads=24 =====
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=24&& .\$(EXE)
	echo ===== KJI (k-j-i) with dynamic=false, proc_bind=close, places=cores, threads=24 =====
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=24&& .\$(KJI_EXE)
	echo ===== IKJ (i-k-j) with dynamic=false, proc_bind=close, places=cores, threads=24 =====
	set OMP_DYNAMIC=FALSE&& set OMP_PROC_BIND=close&& set OMP_PLACES=cores&& set OMP_NUM_THREADS=24&& .\$(IKJ_EXE)

exp-linux:
	echo "===== BASE (i-j-k) with dynamic=false, proc_bind=close, places=cores, threads=12 ====="
	OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=12 ./$(EXE)
	echo "===== KJI (k-j-i) with dynamic=false, proc_bind=close, places=cores, threads=12 ====="
	OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=12 ./$(KJI_EXE)
	echo "===== IKJ (i-k-j) with dynamic=false, proc_bind=close, places=cores, threads=12 ====="
	OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=12 ./$(IKJ_EXE)
	echo "===== BASE (i-j-k) with dynamic=false, proc_bind=close, places=cores, threads=24 ====="
	OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=24 ./$(EXE)
	echo "===== KJI (k-j-i) with dynamic=false, proc_bind=close, places=cores, threads=24 ====="
	OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=24 ./$(KJI_EXE)
	echo "===== IKJ (i-k-j) with dynamic=false, proc_bind=close, places=cores, threads=24 ====="
	OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=24 ./$(IKJ_EXE)