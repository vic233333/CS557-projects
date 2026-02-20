#include "Timer.h"
#include "Laplacian.h"

#include <omp.h>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[])
{
    // Print OpenMP info
#pragma omp parallel
    {
#pragma omp single
        {
            int nThreads = omp_get_num_threads();
            int threadID = omp_get_thread_num();
            std::cout << "Hello from thread " << threadID << " out of " << nThreads << " threads." << std::endl;
        }
    }

    using array_t = float (&)[XDIM][YDIM][ZDIM];

    float* uRaw = new float [XDIM * YDIM * ZDIM];
    float* LuRaw = new float [XDIM * YDIM * ZDIM];
    array_t u = reinterpret_cast<array_t>(*uRaw);
    array_t Lu = reinterpret_cast<array_t>(*LuRaw);

    Timer timer;

    for (int test = 1; test <= 10; test++)
    {
        std::cout << "Running test iteration " << std::setw(2) << test << " ";
        timer.Start();
        ComputeLaplacian(u, Lu);
        timer.Stop("Elapsed time : ");
    }


    return 0;
}
