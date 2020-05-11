import numpy as np
from constants import *

# Define 2D arrays that will store data for electron density, derivatives of e_den, and x/z
dedendz = np.zeros((nx, nz), order='F')    # Backwards, because it is transposed later
dedendx = np.zeros((nx, nz), order='F')

# Define 2D arrays of x and z spatial coordinates
x = np.zeros((nx, nz), order='F')
z = np.zeros((nx, nz), order='F')

for jj in range(nz):
    x[:, jj] = np.linspace(xmin, xmax, nx)

for ii in range(nx):
    z[ii, :] = np.linspace(zmin, zmax, nz)

eden = np.zeros((nx, nz), order='F')

for ii in range(nx):
    for jj in range(nz):
        eden[ii, jj] = max(0.0, ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (x[ii, jj] - xmin) + (0.1 * ncrit))

for ii in range(nx - 1):
    for jj in range(nz - 1):
        dedendx[ii, jj] = (eden[ii + 1, jj] - eden[ii, jj]) / (x[ii + 1, jj] - x[ii, jj])

dedendx[nx - 1, :] = dedendx[nx - 2, :]

# print(dedendx)

# =============================================================================================

aa = np.linspace(xmin, xmax, nx)
bb = np.linspace(zmin, zmax, nz)
xx, zz = np.meshgrid(aa, bb)

eden2 = ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (xx - xmin) + (0.1 * ncrit)
eden2[eden2 < 0.0] = 0.0

dx_test = np.zeros((nz, nx))

for ii in range(nx - 1):
    for jj in range(nz - 1):
        dx_test[jj, ii] = (eden2[jj, ii + 1] - eden2[jj, ii]) / (xx[jj, ii + 1] - xx[jj, ii])

dx_test[:, nx - 1] = dx_test[:, nx - 2]

assert(np.allclose(dx_test.T, dedendx))