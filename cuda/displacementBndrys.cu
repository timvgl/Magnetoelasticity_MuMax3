#include "stencil.h"
#include <stdint.h>
#include <stdio.h>

// Map extended index to interior index by clamping.
// This is used for the tangential (non-normal) components.
__device__ int mapInteriorIndex(int iExt, int N, bool pbcFlag) {
    // If periodic, no ghost layer is present.
    // Otherwise, subtract one.
    int i = pbcFlag ? iExt : (iExt - 1);
    if (i < 0)  i = 0;
    if (i >= N) i = N - 1;
    return i;
}

/**
 * createExtended3DField:
 * Copies a 3D displacement field (ux, uy, uz) of size (Nx, Ny, Nz) into
 * an extended field (uxExt, uyExt, uzExt) whose size is determined by
 * the periodic boundary condition (PBC) flags.
 *
 *  - If PBCx == true, then the x-dimension is not extended (no ghost nodes).
 *  - If PBCx == false, then the x-dimension is extended by 2 (one ghost cell on each side).
 *  - The same logic applies to y and z with PBCy and PBCz.
 *
 * Thus, NxExt = Nx + (PBCx ? 0 : 2),
 *      NyExt = Ny + (PBCy ? 0 : 2),
 *      NzExt = Nz + (PBCz ? 0 : 2).
 *
 * To enforce zero stress, we want the normal second derivative (and hence normal stress)
 * to vanish, and the tangential (shear) stress to vanish. For a given face, we fill:
 *
 *   - The normal displacement component with an extrapolation:
 *         ghost = 2 * u(boundary) - u(adjacent)
 *   - The tangential components with a simple clamping (copy of the boundary value).
 *
 * In the following, for each displacement component we check whether the ghost cell
 * lies on a boundary in the corresponding normal direction. (For example, for ux the
 * x-direction is normal, while y and z are tangential.)
 *
 * Inputs:
 *  - ux, uy, uz: original displacement fields (size Nx*Ny*Nz).
 *  - Nx, Ny, Nz: dimensions of the original field.
 *  - PBC: a bitmask (bits 1,2,4 for x,y,z) indicating periodic boundaries.
 *
 * Outputs:
 *  - uxExt, uyExt, uzExt: extended displacement fields with size NxExt*NyExt*NzExt,
 *    allocated by the user.
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
    // Extended dimensions depend on whether there is a ghost layer or not.
    int NxExt = Nx + (PBCx ? 0 : 2);
    int NyExt = Ny + (PBCy ? 0 : 2);
    int NzExt = Nz + (PBCz ? 0 : 2);

    // Compute extended grid coordinates.
    int xExt = blockIdx.x * blockDim.x + threadIdx.x;
    int yExt = blockIdx.y * blockDim.y + threadIdx.y;
    int zExt = blockIdx.z * blockDim.z + threadIdx.z;

    if (xExt >= NxExt || yExt >= NyExt || zExt >= NzExt)
        return;

    int newIdx = index(xExt, yExt, zExt, NxExt, NyExt, NzExt);

    float finalUx, finalUy, finalUz;

    // --- For the x-component (normal in x, tangential in y,z) ---
    if (!PBCx) {
        if (xExt == 0) {
            // Lower x ghost: use extrapolation for ux.
            int y_phys = mapInteriorIndex(yExt, Ny, PBCy);
            int z_phys = mapInteriorIndex(zExt, Nz, PBCz);
            int idx0 = index(0, y_phys, z_phys, Nx, Ny, Nz);
            int idx1 = (Nx > 1 ? index(1, y_phys, z_phys, Nx, Ny, Nz) : idx0);
            finalUx = 2.0f * ux[idx0] - ux[idx1];
        } else if (xExt == NxExt - 1) {
            // Upper x ghost: extrapolate for ux.
            int y_phys = mapInteriorIndex(yExt, Ny, PBCy);
            int z_phys = mapInteriorIndex(zExt, Nz, PBCz);
            int idx0 = index(Nx - 1, y_phys, z_phys, Nx, Ny, Nz);
            int idx1 = (Nx > 1 ? index(Nx - 2, y_phys, z_phys, Nx, Ny, Nz) : idx0);
            finalUx = 2.0f * ux[idx0] - ux[idx1];
        } else {
            // Interior in x: simply map the index.
            int x_phys = xExt - 1;
            int y_phys = mapInteriorIndex(yExt, Ny, PBCy);
            int z_phys = mapInteriorIndex(zExt, Nz, PBCz);
            finalUx = ux[index(x_phys, y_phys, z_phys, Nx, Ny, Nz)];
        }
    } else {
        // Periodic: no ghost layer.
        finalUx = ux[index(xExt, yExt, zExt, Nx, Ny, Nz)];
    }

    // --- For the y-component (normal in y, tangential in x,z) ---
    if (!PBCy) {
        if (yExt == 0) {
            int x_phys = mapInteriorIndex(xExt, Nx, PBCx);
            int z_phys = mapInteriorIndex(zExt, Nz, PBCz);
            int idx0 = index(x_phys, 0, z_phys, Nx, Ny, Nz);
            int idx1 = (Ny > 1 ? index(x_phys, 1, z_phys, Nx, Ny, Nz) : idx0);
            finalUy = 2.0f * uy[idx0] - uy[idx1];
        } else if (yExt == NyExt - 1) {
            int x_phys = mapInteriorIndex(xExt, Nx, PBCx);
            int z_phys = mapInteriorIndex(zExt, Nz, PBCz);
            int idx0 = index(x_phys, Ny - 1, z_phys, Nx, Ny, Nz);
            int idx1 = (Ny > 1 ? index(x_phys, Ny - 2, z_phys, Nx, Ny, Nz) : idx0);
            finalUy = 2.0f * uy[idx0] - uy[idx1];
        } else {
            int y_phys = yExt - 1;
            int x_phys = mapInteriorIndex(xExt, Nx, PBCx);
            int z_phys = mapInteriorIndex(zExt, Nz, PBCz);
            finalUy = uy[index(x_phys, y_phys, z_phys, Nx, Ny, Nz)];
        }
    } else {
        finalUy = uy[index(xExt, yExt, zExt, Nx, Ny, Nz)];
    }

    // --- For the z-component (normal in z, tangential in x,y) ---
    if (!PBCz) {
        if (zExt == 0) {
            int x_phys = mapInteriorIndex(xExt, Nx, PBCx);
            int y_phys = mapInteriorIndex(yExt, Ny, PBCy);
            int idx0 = index(x_phys, y_phys, 0, Nx, Ny, Nz);
            int idx1 = (Nz > 1 ? index(x_phys, y_phys, 1, Nx, Ny, Nz) : idx0);
            finalUz = 2.0f * uz[idx0] - uz[idx1];
        } else if (zExt == NzExt - 1) {
            int x_phys = mapInteriorIndex(xExt, Nx, PBCx);
            int y_phys = mapInteriorIndex(yExt, Ny, PBCy);
            int idx0 = index(x_phys, y_phys, Nz - 1, Nx, Ny, Nz);
            int idx1 = (Nz > 1 ? index(x_phys, y_phys, Nz - 2, Nx, Ny, Nz) : idx0);
            finalUz = 2.0f * uz[idx0] - uz[idx1];
        } else {
            int z_phys = zExt - 1;
            int x_phys = mapInteriorIndex(xExt, Nx, PBCx);
            int y_phys = mapInteriorIndex(yExt, Ny, PBCy);
            finalUz = uz[index(x_phys, y_phys, z_phys, Nx, Ny, Nz)];
        }
    } else {
        finalUz = uz[index(xExt, yExt, zExt, Nx, Ny, Nz)];
    }

    // Write the computed extended values.
    uxExt[newIdx] = finalUx;
    uyExt[newIdx] = finalUy;
    uzExt[newIdx] = finalUz;
}
