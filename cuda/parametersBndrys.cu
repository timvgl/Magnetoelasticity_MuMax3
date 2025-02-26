#include "stencil.h"
#include <stdint.h>

// Extended index back to the original index.
// If PBC is enabled in a dimension, no ghost cell is present so no shift is applied.
// Otherwise, subtract 1 and then clamp to [0, N-1] to copy the boundary value.
__device__ int mapIndex(int iExt, int N, bool pbcFlag) {
    int i = pbcFlag ? iExt : (iExt - 1);
    if (i < 0) {
        i = 0;
    }
    if (i >= N) {
        i = N - 1;
    }
    return i;
}

/**
 * extendParameterField:
 * Copies a 3D parameter field (params) of size (Nx, Ny, Nz) into an extended field
 * (paramsExt) whose size is determined by the periodic boundary condition (PBC) flags.
 *
 *  - If PBCx == true, then the x-dimension is not extended (no ghost nodes).
 *  - If PBCx == false, then the x-dimension is extended by 2 (one ghost cell on each side).
 *  - The same applies for the y and z dimensions with PBCy and PBCz.
 *
 * Thus, NxExt = Nx + (PBCx ? 0 : 2),
 *      NyExt = Ny + (PBCy ? 0 : 2),
 *      NzExt = Nz + (PBCz ? 0 : 2).
 *
 * The ghost cells (if present) are filled by copying the nearest interior value,
 * which approximates a zero-gradient (Neumann) condition.
 *
 * Inputs:
 *  - params: original parameter field (size Nx*Ny*Nz).
 *  - Nx, Ny, Nz: dimensions of the original field.
 *  - PBCx, PBCy, PBCz: flags indicating periodic boundaries in each dimension.
 *
 * Output:
 *  - paramsExt: extended parameter field with size NxExt*NyExt*NzExt.
 */
 extern "C" __global__ void
 extendParameterField(
    float* __restrict__ paramsExt,
    float* __restrict__ params,
    int Nx, int Ny, int Nz,
    uint8_t PBC)
{
    // Compute extended dimensions based on PBC flags
    int NxExt = Nx + (PBCx ? 0 : 2);
    int NyExt = Ny + (PBCy ? 0 : 2);
    int NzExt = Nz + (PBCz ? 0 : 2);

    // Compute the (xExt, yExt, zExt) coordinates in the extended grid
    int xExt = blockIdx.x * blockDim.x + threadIdx.x;
    int yExt = blockIdx.y * blockDim.y + threadIdx.y;
    int zExt = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if thread is within extended array bounds
    if (xExt >= NxExt || yExt >= NyExt || zExt >= NzExt)
        return;

    int x = mapIndex(xExt, Nx, PBCx);
    int y = mapIndex(yExt, Ny, PBCy);
    int z = mapIndex(zExt, Nz, PBCz);

    // Compute linear indices for original and extended arrays.
    int oldIdx = index(x, y, z, Nx, Ny, Nz);
    int newIdx = index(xExt, yExt, zExt, NxExt, NyExt, NzExt);

    // Copy the parameter value from the original field to the extended field.
    paramsExt[newIdx] = params[oldIdx];
}
