import matplotlib.pyplot as plt
from constants import *
from numpy import max as npmax

def plot_everything(grid, intensity_sum, variable1, a0_variable, keep_open=False):
    cmap = 'jet'
    plt.figure()
    plt.pcolormesh(grid.z, grid.x, grid.eden / ncrit, cmap=cmap)
    plt.plot(grid.z - (dz / 2), grid.x - (dx / 2), 'k--')
    plt.plot(grid.x - (dx / 2), grid.z - (dz / 2), 'k--')

    plt.plot(grid.z - (dz / 2), grid.x + (dx / 2), 'k--')
    plt.plot(grid.x + (dx / 2), grid.z - (dz / 2), 'k--')

    plt.plot(grid.z + (dz / 2), grid.x - (dx / 2), 'k--')
    plt.plot(grid.x - (dx / 2), grid.z + (dz / 2), 'k--')

    plt.plot(grid.z + (dz / 2), grid.x + (dx / 2), 'k--')
    plt.plot(grid.x + (dx / 2), grid.z + (dz / 2), 'k--')

    plt.plot(grid.z, grid.x, 'k--')
    plt.plot(grid.x, grid.z, 'k--')

    plt.colorbar()

    plt.xlabel('z (cm)')
    plt.ylabel('x (cm)')
    plt.title('n_e_/n_crit_')

    plt.show(block=False)

    '''Plot the cumulative energy deposited to the array edep, which shares the dimensions of grid.x, grid.z, grid.eden, dedendz, etc.'''
    for b in range(nbeams):
        for n in range(nrays):
            finalt = grid.finalts[b, n]
            plt.plot(grid.mysaved_z[b, n, :finalt], grid.mysaved_x[b, n, :finalt], 'm')

    plt.show(block=False)

    plt.figure()
    clo = 0.0
    chi = npmax(intensity_sum)
    plt.pcolormesh(grid.z, grid.x, intensity_sum, cmap=cmap, vmin=clo, vmax=chi)
    plt.colorbar()
    plt.xlabel('z (cm)')
    plt.ylabel('x (cm)')
    plt.title('Overlapped intensity')
    plt.show(block=False)

    plt.figure()
    plt.pcolormesh(grid.z, grid.x, variable1, cmap=cmap, vmin=0.0, vmax=0.021)
    plt.colorbar()
    plt.xlabel('z (cm)')
    plt.ylabel('x (cm)')
    plt.title('Total original field amplitude (a0)')
    plt.show(block=False)

    plt.figure()
    plt.pcolormesh(grid.z, grid.x, a0_variable, cmap=cmap, vmin=0.0, vmax=0.021)
    plt.colorbar()
    plt.xlabel('z (cm)')
    plt.ylabel('x (cm)')
    plt.title('Total CBET new field amplitude (a0)')
    plt.show(block=False)

    plt.figure()
    plt.plot(grid.x[0, :], a0_variable[1, :], ',-b')
    plt.plot(grid.x[0, :], a0_variable[nz - 2, :], ',-r')
    plt.plot(grid.x[0, :], a0_variable[nz // 2, :], ',-g')
    plt.xlabel('x (cm)')
    plt.ylabel('a0')
    plt.title('a0(x) at z_min, z_0, z_max')
    plt.grid(linestyle='--')
    plt.show(block=False)

    plt.figure()
    plt.plot(grid.z[:, 0], a0_variable[:, 1], ',-b')
    plt.plot(grid.z[:, 0], a0_variable[:, nx - 2], ',-r')
    plt.plot(grid.z[:, 0], a0_variable[:, nx // 2], ',-g')
    plt.xlabel('z (cm)')
    plt.ylabel('a0')
    plt.title('a0(z) at x_min, x_0, x_max')
    plt.grid(linestyle='--')
    plt.show(block=keep_open)
