#include "Laplacian.h"
#include "Parameters.h"
#include "PointwiseOps.h"
#include "Reductions.h"
#include "Utilities.h"
#include "Timer.h"

#include <iostream>

// External timer declarations for CG algorithm profiling
extern Timer timerCG; // Entire CG algorithm
extern Timer timerLaplacian_line2; // ComputeLaplacian(x, z) - outside loop
extern Timer timerSaxpy_line2; // Saxpy(z, f, r, -1) - outside loop
extern Timer timerNorm_line2; // Norm(r) - outside loop
extern Timer timerCopy_line4; // Copy(r, p) - outside loop
extern Timer timerInnerProduct_line4; // InnerProduct(p, r) - outside loop
extern Timer timerLaplacian_line6; // ComputeLaplacian(p, z) - inside loop (MOST IMPORTANT)
extern Timer timerInnerProduct_line6; // InnerProduct(p, z) - inside loop
extern Timer timerSaxpy_line8; // Saxpy(z, r, r, -alpha) - inside loop
extern Timer timerNorm_line8; // Norm(r) - inside loop
extern Timer timerCopy_line13; // Copy(r, z) - inside loop
extern Timer timerInnerProduct_line13; // InnerProduct(z, r) - inside loop
extern Timer timerSaxpy_line16a; // Saxpy(p, x, x, alpha) - inside loop
extern Timer timerSaxpy_line16b; // Saxpy(p, r, p, beta) - inside loop

void ConjugateGradients(
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const bool writeIterations)
{
    timerCG.Restart();

    // Algorithm : Line 2
    timerLaplacian_line2.Restart();
    ComputeLaplacian(x, z);
    timerLaplacian_line2.Pause();

    timerSaxpy_line2.Restart();
    Saxpy(z, f, r, -1);
    timerSaxpy_line2.Pause();

    timerNorm_line2.Restart();
    float nu = Norm(r);
    timerNorm_line2.Pause();

    // Algorithm : Line 3
    if (nu < nuMax)
    {
        timerCG.Pause();
        return;
    }

    // Algorithm : Line 4
    timerCopy_line4.Restart();
    Copy(r, p);
    timerCopy_line4.Pause();

    timerInnerProduct_line4.Restart();
    float rho = InnerProduct(p, r);
    timerInnerProduct_line4.Pause();

    // Beginning of loop from Line 5
    for (int k = 0;; k++)
    {
        std::cout << "Residual norm (nu) after " << k << " iterations = " << nu << std::endl;

        // Algorithm : Line 6
        timerLaplacian_line6.Restart();
        ComputeLaplacian(p, z);
        timerLaplacian_line6.Pause();

        timerInnerProduct_line6.Restart();
        float sigma = InnerProduct(p, z);
        timerInnerProduct_line6.Pause();

        // Algorithm : Line 7
        float alpha = rho / sigma;

        // Algorithm : Line 8
        timerSaxpy_line8.Restart();
        Saxpy(z, r, r, -alpha);
        timerSaxpy_line8.Pause();

        timerNorm_line8.Restart();
        nu = Norm(r);
        timerNorm_line8.Pause();

        // Algorithm : Lines 9-12
        if (nu < nuMax || k == kMax)
        {
            Saxpy(p, x, x, alpha);

            std::cout << "Conjugate Gradients terminated after " << k << " iterations; residual norm (nu) = " << nu <<
                std::endl;
            if (writeIterations) WriteAsImage("x", x, k, 0, 127);

            timerCG.Pause();
            return;
        }

        // Algorithm : Line 13
        timerCopy_line13.Restart();
        Copy(r, z);
        timerCopy_line13.Pause();

        timerInnerProduct_line13.Restart();
        float rho_new = InnerProduct(z, r);
        timerInnerProduct_line13.Pause();

        // Algorithm : Line 14
        float beta = rho_new / rho;

        // Algorithm : Line 15
        rho = rho_new;

        // Algorithm : Line 16
        timerSaxpy_line16a.Restart();
        Saxpy(p, x, x, alpha);
        timerSaxpy_line16a.Pause();

        timerSaxpy_line16b.Restart();
        Saxpy(p, r, p, beta);
        timerSaxpy_line16b.Pause();

        if (writeIterations) WriteAsImage("x", x, k, 0, 127);
    }
}
