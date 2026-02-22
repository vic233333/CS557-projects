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
