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

float LaplacianAndDot(
    const float (&p)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM])
{
    double result = 0.0;

#pragma omp parallel for reduction(+:result)
    for (int i = 1; i < XDIM-1; i++)
    for (int j = 1; j < YDIM-1; j++)
    for (int k = 1; k < ZDIM-1; k++) {
        const float lp =
            -6 * p[i][j][k]
        + p[i+1][j][k] + p[i-1][j][k]
        + p[i][j+1][k] + p[i][j-1][k]
        + p[i][j][k+1] + p[i][j][k-1];
        z[i][j][k] = lp; // Write the Laplacian to memory as required by the algorithm
        result += static_cast<double>(p[i][j][k]) * static_cast<double>(lp);
    }

    return static_cast<float>(result);
}