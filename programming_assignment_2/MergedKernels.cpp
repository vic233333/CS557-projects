#include "MergedKernels.h"

#include <algorithm>
#include <cmath>

float SaxpyAndNorm(
    const float (&z)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    const float alpha)
{
    float result = 0.f;

#pragma omp parallel for reduction(max:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++) {
        r[i][j][k] = -alpha * z[i][j][k] + r[i][j][k];
        result = std::max(result, std::abs(r[i][j][k]));
    }

    return result;
}

void DoubleSaxpy(
    float (&x)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    const float alpha,
    const float beta)
{
#pragma omp parallel for
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++) {
        float p_val = p[i][j][k]; // Read p[i][j][k] once, use for both updates
        x[i][j][k] = alpha * p_val + x[i][j][k];
        p[i][j][k] = beta  * p_val + r[i][j][k];
    }
}