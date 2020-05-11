import cupy as cp
import numpy as np
from constants import *
from grid_space import GridSpace
from plotter import plot_everything
from time import monotonic


def main():
    grid = GridSpace()

    # Parameters for first beam
    beam = 0
    x0 = np.full(nrays, xmin - (dt / courant_mult * c * 0.5))
    z0 = np.linspace(beam_min_z, beam_max_z, nrays) + offset - (dz / 2) - (dt / courant_mult * c * 0.5)
    kx0 = np.full(nrays, 1.0)
    kz0 = np.full(nrays, -0.1)
    # Launch rays for first beam
    grid.beam_me_up(beam, x0, z0, kx0, kz0)

    # Parameters for second beam
    beam = 1
    z0 = np.full(nrays, zmin - (dt / courant_mult * c * 0.5))
    x0 = np.linspace(beam_min_z, beam_max_z, nrays) - (dx / 2) - (dt / courant_mult * c * 0.5)
    kx0 = np.zeros(nrays)
    kz0 = np.ones(nrays)
    # Launch rays for second beam
    grid.beam_me_up(beam, x0, z0, kx0, kz0)

    dkx = grid.crosses_x[:, :, 1:] - grid.crosses_x[:, :, :-1]
    dkz = grid.crosses_z[:, :, 1:] - grid.crosses_z[:, :, :-1]
    dkmag = np.sqrt(dkx**2 + dkz**2)

    i_b1 = np.copy(grid.edep[0, :nz, :nx])
    i_b2 = np.copy(grid.edep[1, :nz, :nx])

    W1 = np.sqrt(1 - grid.eden / ncrit) / rays_per_zone
    W2 = np.sqrt(1 - grid.eden / ncrit) / rays_per_zone

    W1_new = np.copy(W1)
    W2_new = np.copy(W2)

    for bb in range(nbeams - 1):
        grid.calculate_gain(bb, dkx, dkz, dkmag, W1, W2, W1_new, W2_new, i_b1)

    i_b1_new = np.copy(i_b1)
    i_b2_new = np.copy(i_b2)

    for bb in range(nbeams - 1):
        grid.calculate_intensity(bb, W1, W2, W1_new, W2_new, i_b1, i_b2, i_b1_new, i_b2_new)

    intensity_sum = np.sum(grid.edep[:, :nz, :nx], axis=0)
    variable1 = 8.53e-10 * np.sqrt(i_b1 + i_b2 + 1.0e-10) * (1.053 / 3.0)
    i_b1_new[i_b1_new < 1.0e-10] = 1.0e-10
    i_b2_new[i_b2_new < 1.0e-10] = 1.0e-10
    a0_variable = 8.53e-10 * np.sqrt(i_b1_new + i_b2_new + 1.0e-10) * (1.053 / 3.0)

    plot_everything(grid, intensity_sum, variable1, a0_variable, keep_open=True)

if __name__ == '__main__':
    start = monotonic()
    main()
    print(f'Total time: {monotonic() - start} seconds')
