import numpy as np
from constants import *

class Ray:
    def __init__(self, grid, beam, n, uray_0, x_init, z_init, kx_init, kz_init):
        self.finalt = 0
        self.rayx = 0
        self.rayz = 0

        self.my_x = np.zeros(nt)
        self.my_x[0] = x_init

        self.my_z = np.zeros(nt)
        self.my_z[0] = z_init

        self.marking_x = np.zeros(nt, dtype=np.int32)
        self.marking_z = np.zeros(nt, dtype=np.int32)

        self.my_vx = np.zeros(nt)
        self.my_vz = np.zeros(nt)

        self.launch_ray(grid, beam, n, uray_0, kx_init, kz_init)

    def get_finalt(self):
        return self.finalt

    def get_rayx(self):
        return self.rayx

    def get_rayz(self):
        return self.rayz

    def launch_ray(self, grid, beam, n, uray_0, kx_init, kz_init):
        for xx in range(nx):
            if ((-0.5 - 1.0e-10) * dx) <= (self.my_x[0] - grid.x[0, xx]) <= ((0.5 + 1.0e-10) * dx):
                thisx_0 = xx
                break  # "breaks" out of the xx loop once the if statement condition is met.

        for zz in range(nz):
            if ((-0.5 - 1.0e-10) * dz) <= (self.my_z[0] - grid.z[zz, 0]) <= ((0.5 + 1.0e-10) * dz):
                thisz_0 = zz
                break  # "breaks" out of the zz loop once the if statement condition is met.

        knorm = np.sqrt(kx_init**2 + kz_init**2)

        mykx = np.sqrt((omega**2 - grid.wpe[thisz_0, thisx_0]**2) / c**2) * (kx_init / knorm)
        mykz = np.sqrt((omega**2 - grid.wpe[thisz_0, thisx_0]**2) / c**2) * (kz_init / knorm)

        self.my_vx[0] = (c ** 2) * mykx / omega
        self.my_vz[0] = (c ** 2) * mykz / omega

        self.marking_x[0] = thisx_0
        self.marking_z[0] = thisz_0

        numcrossing = 1

        for tt in range(1, nt):
            self.my_vz[tt] = self.my_vz[tt - 1] - (c ** 2) / (2.0 * ncrit) * grid.dedendz[thisz_0, thisx_0] * dt
            self.my_vx[tt] = self.my_vx[tt - 1] - (c ** 2) / (2.0 * ncrit) * grid.dedendx[thisz_0, thisx_0] * dt

            self.my_x[tt] = self.my_x[tt - 1] + self.my_vx[tt] * dt
            self.my_z[tt] = self.my_z[tt - 1] + self.my_vz[tt] * dt

            search_index_x = 1
            search_index_z = 1

            thisx_m = max(0, thisx_0 - search_index_x)
            thisx_p = min(nx, thisx_0 + search_index_x)

            thisz_m = max(0, thisz_0 - search_index_z)
            thisz_p = min(nz, thisz_0 + search_index_z)

            for xx in range(thisx_m, thisx_p + 1):  # Determines current x index for the position
                if xx >= nx:
                    xx = nx - 1
                if (dx * (0.5 + 1.0e-10)) >= (self.my_x[tt] - grid.x[0, xx]) >= (-1 * (0.5 + 1.0e-10) * dx):
                    thisx = xx
                    break


            for zz in range(thisz_m, thisz_p + 1):  # Determines current z index for the position
                if zz >= nz:
                    zz = nz - 1
                if (dz * (0.5 + 1.0e-10)) >= (self.my_z[tt] - grid.z[zz, 0]) >= (-1 * (0.5 + 1.0e-10) * dz):
                    thisz = zz
                    break

            linex = [self.my_x[tt - 1], self.my_x[tt]]
            linez = [self.my_z[tt - 1], self.my_z[tt]]

            lastx = 10000
            lastz = 10000

            for xx in range(thisx_m, thisx_p):
                currx = grid.x[0, xx] - (dx / 2)

                if (self.my_x[tt] > currx >= self.my_x[tt - 1]) or (self.my_x[tt] < currx <= self.my_x[tt - 1]):
                    crossx = np.interp(currx, linex, linez)
                    if abs(crossx - lastz) > 1e-20:
                        grid.crosses_x[beam, n, numcrossing] = currx
                        grid.crosses_z[beam, n, numcrossing] = crossx

                        if (xmin - dx / 2) <= self.my_x[tt] <= (xmax + dx / 2):
                            grid.boxes[beam, n, numcrossing, :] = [thisz, thisx]

                        lastx = currx
                        numcrossing += 1
                        break

            for zz in range(thisz_m, thisz_p):
                currz = grid.z[zz, 0] - (dz / 2)
                if (self.my_z[tt] > currz >= self.my_z[tt - 1]) or (self.my_z[tt] < currz <= self.my_z[tt - 1]):
                    crossz = np.interp(currz, linez, linex)
                    if abs(crossz - lastx) > 1.0e-20:
                        grid.crosses_x[beam, n, numcrossing] = crossz
                        grid.crosses_z[beam, n, numcrossing] = currz

                        if (zmin - dz / 2) <= self.my_z[tt] <= (zmax + dz / 2):
                            grid.boxes[beam, n, numcrossing, :] = [thisz, thisx]

                    lastz = currz
                    numcrossing += 1
                    break

            thisx_0 = thisx
            thisz_0 = thisz

            self.marking_x[tt] = thisx
            self.marking_z[tt] = thisz

            for ss in range(numstored):
                if grid.marked[beam, ss, thisz, thisx] == 0:
                    grid.marked[beam, ss, thisz, thisx] = n
                    grid.present[beam, thisz, thisx] += 1
                    break

            xp = (self.my_x[tt] - (grid.x[thisz, thisx] + dx / 2)) / dx
            zp = (self.my_z[tt] - (grid.z[thisz, thisx] + dz / 2)) / dz

            dl = abs(zp)
            dm = abs(xp)
            a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
            a2 = (1.0 - dl) * dm  # green : (x+1, z)
            a3 = dl * (1.0 - dm)  # yellow : (x, z+1)
            a4 = dl * dm  # red : (x+1, z+1)

            if xp >= 0 and zp >= 0:
                grid.edep[beam, thisz + 1, thisx + 1] += a1 * uray_0  # blue
                grid.edep[beam, thisz + 1, thisx + 2] += a2 * uray_0  # green
                grid.edep[beam, thisz + 2, thisx + 1] += a3 * uray_0  # yellow
                grid.edep[beam, thisz + 2, thisx + 2] += a4 * uray_0  # red
            elif xp < 0 and zp >= 0:
                grid.edep[beam, thisz + 1, thisx + 1] += a1 * uray_0  # blue
                grid.edep[beam, thisz + 1, thisx]     += a2 * uray_0  # green
                grid.edep[beam, thisz + 2, thisx + 1] += a3 * uray_0  # yellow
                grid.edep[beam, thisz + 2, thisx]     += a4 * uray_0  # red
            elif xp >= 0 and zp < 0:
                grid.edep[beam, thisz + 1, thisx + 1] += a1 * uray_0  # blue
                grid.edep[beam, thisz + 1, thisx + 2] += a2 * uray_0  # green
                grid.edep[beam, thisz, thisx + 1]     += a3 * uray_0  # yellow
                grid.edep[beam, thisz, thisx + 2]     += a4 * uray_0  # red
            elif xp < 0 and zp < 0:
                grid.edep[beam, thisz + 1, thisx + 1] += a1 * uray_0  # blue
                grid.edep[beam, thisz + 1, thisx]     += a2 * uray_0  # green
                grid.edep[beam, thisz, thisx + 1]     += a3 * uray_0  # yellow
                grid.edep[beam, thisz, thisx]         += a4 * uray_0  # red
            else:
                print(f'xp is {xp}, zp is {zp}')
                print('***** ERROR in interpolation of laser deposition grid!! *****')
                break

                # This will cause the code to stop following the ray once it escapes the extent of the plasma
            if self.my_x[tt] < (xmin - (dx / 2.0)) or self.my_x[tt] > (xmax + (dx / 2.0)):
                self.finalt = tt - 1
                self.rayx = self.my_x[:self.finalt]
                self.rayz = self.my_z[:self.finalt]
                break
            elif self.my_z[tt] < (zmin - (dz / 2.0)) or self.my_z[tt] > (zmax + (dz / 2.0)):
                self.finalt = tt - 1
                self.rayx = self.my_x[:self.finalt]
                self.rayz = self.my_z[:self.finalt]
                break


