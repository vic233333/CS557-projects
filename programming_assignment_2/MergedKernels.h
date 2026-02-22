#pragma once

#include "Parameters.h"

// Merged: Saxpy(z, r, r, -alpha) + Norm(r)
// Computes r = -alpha*z + r, returns max|r|
float SaxpyAndNorm(
    const float (&z)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float alpha);