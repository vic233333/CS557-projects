#include "ConjugateGradients.h"
#include "Timer.h"
#include "Utilities.h"

// Timer declarations for CG algorithm profiling
Timer timerCG; // Entire CG algorithm
Timer timerLaplacian_line2; // ComputeLaplacian(x, z) - outside loop
Timer timerSaxpy_line2; // Saxpy(z, f, r, -1) - outside loop
Timer timerNorm_line2; // Norm(r) - outside loop
Timer timerCopy_line4; // Copy(r, p) - outside loop
Timer timerInnerProduct_line4; // InnerProduct(p, r) - outside loop
Timer timerLaplacianAndDot_line6; // LaplacianAndDot(p, z) - inside loop (merged version)
Timer timerLaplacian_line6; // ComputeLaplacian(p, z) - inside loop (MOST IMPORTANT)
Timer timerInnerProduct_line6; // InnerProduct(p, z) - inside loop
Timer timerSaxpyAndNorm_line8; // SaxpyAndNorm(z, r, alpha) - inside loop (merged version)
Timer timerSaxpy_line8; // Saxpy(z, r, r, -alpha) - inside loop
Timer timerNorm_line8; // Norm(r) - inside loop
Timer timerCopy_line13; // Copy(r, z) - inside loop
Timer timerInnerProduct_line13; // InnerProduct(z, r) - inside loop
Timer timerDoubleSaxpy_line16; // DoubleSaxpy(x, r, p, alpha, beta) - inside loop (merged version)
Timer timerSaxpy_line16a; // Saxpy(p, x, x, alpha) - inside loop
Timer timerSaxpy_line16b; // Saxpy(p, r, p, beta) - inside loop

int main(int argc, char* argv[])
{
    using array_t = float (&)[XDIM][YDIM][ZDIM];

    float* xRaw = new float [XDIM * YDIM * ZDIM];
    float* fRaw = new float [XDIM * YDIM * ZDIM];
    float* pRaw = new float [XDIM * YDIM * ZDIM];
    float* rRaw = new float [XDIM * YDIM * ZDIM];
    float* zRaw = new float [XDIM * YDIM * ZDIM];

    array_t x = reinterpret_cast<array_t>(*xRaw);
    array_t f = reinterpret_cast<array_t>(*fRaw);
    array_t p = reinterpret_cast<array_t>(*pRaw);
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t z = reinterpret_cast<array_t>(*zRaw);

    // Reset timers before CG
    timerCG.Reset();
    timerLaplacian_line2.Reset();
    timerSaxpy_line2.Reset();
    timerNorm_line2.Reset();
    timerCopy_line4.Reset();
    timerInnerProduct_line4.Reset();
    timerLaplacianAndDot_line6.Reset();
    timerLaplacian_line6.Reset();
    timerInnerProduct_line6.Reset();
    timerSaxpyAndNorm_line8.Reset();
    timerSaxpy_line8.Reset();
    timerNorm_line8.Reset();
    timerCopy_line13.Reset();
    timerInnerProduct_line13.Reset();
    timerDoubleSaxpy_line16.Reset();
    timerSaxpy_line16a.Reset();
    timerSaxpy_line16b.Reset();

#ifdef USE_MERGED
    std::cout << "====== NOW RUNNING MERGED VERSION ======" << std::endl;
#else
    std::cout << "====== NOW RUNNING UNMERGED VERSION ======" << std::endl;
#endif


    // Initialization
    {
        Timer timer;
        timer.Start();
        InitializeProblem(x, f);
        timer.Stop("Initialization : ");
    }

    // Call Conjugate Gradients algorithm
    ConjugateGradients(x, f, p, r, z, false);

#ifdef USE_MERGED
        std::cout << "====== USING MERGED VERSION ======" << std::endl;
#else
        std::cout << "====== USING UNMERGED VERSION ======" << std::endl;
#endif

    // Print timers after CG
    timerCG.Print("Total CG: ");
    timerLaplacian_line2.Print("ComputeLaplacian (line 2): ");
    timerSaxpy_line2.Print("Saxpy (line 2): ");
    timerNorm_line2.Print("Norm (line 2): ");
    timerCopy_line4.Print("Copy (line 4): ");
    timerInnerProduct_line4.Print("InnerProduct (line 4): ");
    timerLaplacianAndDot_line6.Print("LaplacianAndDot (line 6, cumulative): ");
    timerLaplacian_line6.Print("ComputeLaplacian (line 6, cumulative): ");
    timerInnerProduct_line6.Print("InnerProduct (line 6): ");
    timerSaxpyAndNorm_line8.Print("SaxpyAndNorm (line 8): ");
    timerSaxpy_line8.Print("Saxpy (line 8): ");
    timerNorm_line8.Print("Norm (line 8): ");
    timerCopy_line13.Print("Copy (line 13): ");
    timerInnerProduct_line13.Print("InnerProduct (line 13): ");
    timerDoubleSaxpy_line16.Print("DoubleSaxpy (line 16): ");
    timerSaxpy_line16a.Print("Saxpy (line 16a): ");
    timerSaxpy_line16b.Print("Saxpy (line 16b): ");

    // Clean up
    delete[] xRaw;
    delete[] fRaw;
    delete[] pRaw;
    delete[] rRaw;
    delete[] zRaw;

    return 0;
}
