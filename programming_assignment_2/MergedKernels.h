#pragma once

#include "Parameters.h"

// Merged: Saxpy(z, r, r, -alpha) + Norm(r)
// Computes r = -alpha*z + r, returns max|r|
float SaxpyAndNorm(
    const float (&z)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float alpha);

// Merged: Saxpy(p, x, x, alpha) + Saxpy(p, r, p, beta)
// Computes x = alpha*p + x  AND  p = beta*p + r  in one pass
void DoubleSaxpy(
    float (&x)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    const float alpha,
    const float beta);

// Merged: ComputeLaplacian(p, z) + InnerProduct(p, z)
// Computes z = Lp, returns p·z = p·Lp, without writing z to memory
float LaplacianAndDot(
    const float (&p)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM]);

// Merged: ComputeLaplacian(x,z) + Saxpy(z,f,r,-1) + Norm(r)
// Computes r = f - Lx, returns max|r|, z not needed
float LaplacianSaxpyAndNorm(
    const float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM]);