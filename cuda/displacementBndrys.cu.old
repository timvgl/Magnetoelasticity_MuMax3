#include "stencil.h"
#include <stdint.h>

// Map (xExt, yExt, zExt) back to an index in the original arrays (x, y, z).
// If PBC is true in a dimension, we do not have ghost nodes there,
// so x = xExt, y = yExt, z = zExt. If PBC is false, we have 1 ghost cell on each side,
// so we shift by 1 (x = xExt - 1, etc.).
__device__ int mapIndex(int iExt, int N, bool pbcFlag) {
    // If PBC is active, there's no shift (and no ghost layer).
    // If PBC is not active, shift by 1 to remove the ghost offset.
    int i = pbcFlag ? iExt : (iExt - 1);

    // Now clamp the index to [0, N-1] to approximate free boundary (Neumann).
    // If i is already in [0, N-1], it stays the same.
    if (i < 0) {
        i = 0;
    }
    if (i >= N) {
        i = N - 1;
    }

    return i;
}

/**
 * createExtended3DField:
 * Copies a 3D displacement field (ux, uy, uz) of size (Nx, Ny, Nz) into
 * potentially larger fields (uxExt, uyExt, uzExt). The new size is determined by
 * whether or not there is a periodic boundary condition (PBC) in each dimension.
 *
 *  - If PBCx == true, then the x-dimension is not extended (no ghost nodes).
 *  - If PBCx == false, then the x-dimension is extended by 2 (one ghost cell on each side).
 *  - The same logic applies to y and z dimensions using PBCy and PBCz.
 *
 * Thus, NxExt = Nx + (PBCx ? 0 : 2),
 *      NyExt = Ny + (PBCy ? 0 : 2),
 *      NzExt = Nz + (PBCz ? 0 : 2).
 *
 * The ghost nodes (if present) are filled by simply clamping the index to the
 * nearest interior cell, approximating a zero derivative (free boundary).
 * For the dimensions with PBC, no ghost nodes are added at all.
 *
 * Inputs:
 *  - ux, uy, uz:  original displacement fields (size Nx*Ny*Nz).
 *  - Nx, Ny, Nz:  original dimensions.
 *  - PBCx, PBCy, PBCz: flags for periodic boundaries in each dimension.
 *
 * Outputs:
 *  - uxExt, uyExt, uzExt: extended displacement fields with size NxExt*NyExt*NzExt,
 *    allocated by the user. This kernel writes the appropriate values into them.
 */
extern "C" __global__ void
createExtended3DField(
    float* __restrict__ uxExt,
    float* __restrict__ uyExt,
    float* __restrict__ uzExt,
    float* __restrict__ ux,
    float* __restrict__ uy,
    float* __restrict__ uz,
    int Nx, int Ny, int Nz,
    uint8_t PBC)
{
    // Compute the extended dimensions depending on the PBC flags
    int NxExt = Nx + (PBCx ? 0 : 2);
    int NyExt = Ny + (PBCy ? 0 : 2);
    int NzExt = Nz + (PBCz ? 0 : 2);

    // Determine the (xExt, yExt, zExt) coordinates in the extended grid
    int xExt = blockIdx.x * blockDim.x + threadIdx.x;
    int yExt = blockIdx.y * blockDim.y + threadIdx.y;
    int zExt = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if we are within the extended array bounds
    if (xExt >= NxExt || yExt >= NyExt || zExt >= NzExt) {
        return;
    }

    int x = mapIndex(xExt, Nx, PBCx);
    int y = mapIndex(yExt, Ny, PBCy);
    int z = mapIndex(zExt, Nz, PBCz);

    // Compute the linear indices for original and extended arrays
    int oldIdx = index(x, y, z, Nx, Ny, Nz);
    int newIdx = index(xExt, yExt, zExt, NxExt, NyExt, NzExt);

    // Copy the displacement values
    uxExt[newIdx] = ux[oldIdx];
    uyExt[newIdx] = uy[oldIdx];
    uzExt[newIdx] = uz[oldIdx];
}
