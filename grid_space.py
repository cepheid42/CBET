import numpy as np
from constants import *
import launch_ray as lr


class GridSpace:
    def __init__(self):
        aa = np.linspace(xmin, xmax, nx)
        bb = np.linspace(zmin, zmax, nz)
        self.x, self.z = np.meshgrid(aa, bb)

        self.edep = np.zeros((nbeams, nz + 2, nx + 2))

        self.eden = ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (self.x - xmin) + (0.1 * ncrit)
        self.eden[self.eden < 0.0] = 0.0

        # Streamlined u_flow, since Machnum was only used to compute u_flow
        self.u_flow = (((-0.4) - (-2.4)) / (xmax - xmin)) * (self.x - xmin)
        self.u_flow[self.u_flow < 0] = 0.0
        self.u_flow -= 2.4
        self.u_flow *= cs

        self.wpe = np.sqrt(self.eden * 1e6 * e_c ** 2 / (m_e * e_0))

        self.dedendx = np.zeros((nz, nx))
        self.dedendz = np.zeros((nz, nx))

        for j in range(nz - 1):
            for i in range(nx - 1):
                self.dedendx[j, i] = (self.eden[j, i + 1] - self.eden[j, i]) / (self.x[j, i + 1] - self.x[j, i])
                self.dedendz[j, i] = (self.eden[j + 1, i] - self.eden[j, i]) / (self.z[j + 1, i] - self.z[j, i])

        self.dedendz[nz - 1, :] = self.dedendz[nz - 2, :]
        self.dedendx[:, nx - 1] = self.dedendx[:, nx - 2]

        self.finalts = np.zeros((nbeams, nrays), dtype=np.int32)

        self.marked = np.zeros((nbeams, numstored, nz, nx), dtype=np.int32)
        self.boxes = np.zeros((nbeams, nrays, ncrossings, 2), dtype=np.int32)
        self.present = np.zeros((nbeams, nz, nx), dtype=np.int32)

        self.crosses_x = np.zeros((nbeams, nrays, ncrossings))
        self.crosses_z = np.zeros((nbeams, nrays, ncrossings))

        self.mysaved_x = np.zeros((nbeams, nrays, nt))
        self.mysaved_z = np.zeros((nbeams, nrays, nt))

        self.intersections = np.zeros((nz, nx), dtype=np.int32)

    def beam_me_up(self, beam, x0, z0, kx0, kz0):
        for n in range(nrays):
            uray_0 = uray_mult * np.interp(z0[n], phase_x + offset, pow_x)
            dummy = lr.Ray(self, beam, n, uray_0, x0[n], z0[n], kx0[n], kz0[n])

            finalt = dummy.get_finalt()

            self.finalts[beam, n] = finalt
            self.mysaved_x[beam, n, :finalt] = dummy.get_rayx()
            self.mysaved_z[beam, n, :finalt] = dummy.get_rayz()

    def count_intersections(self):
        for xx in range(1, nx):
            for zz in range(1, nz):
                for ss in range(numstored):
                    if self.marked[0, ss, zz, xx] == 0:
                        break
                    else:
                        for sss in range(numstored):
                            if self.marked[1, sss, zz, xx] == 0:
                                break
                            else:
                                self.intersections[zz, xx] += 1

    def calculate_gain(self, bb, dkx, dkz, dkmag, W1, W2, W1_new, W2_new, i_b1):
        for rr1 in range(nrays):
            for cc1 in range(ncrossings):
                if self.boxes[bb, rr1, cc1, 0] == 0 or self.boxes[bb, rr1, cc1, 1] == 0:
                    break

                ix = self.boxes[bb, rr1, cc1, 0]
                iz = self.boxes[bb, rr1, cc1, 1]

                if self.intersections[iz, ix] != 0:
                    nonzeros2 = self.marked[bb + 1, :, iz, ix].nonzero()
                    numrays2 = np.count_nonzero(self.marked[bb + 1, :, iz, ix])

                    marker2 = self.marked[bb + 1, nonzeros2, iz, ix].flatten()

                    rr2 = marker2
                    cc2 = marker2

                    for n2 in range(numrays2):
                        for ccc in range(ncrossings):
                            ix2 = self.boxes[bb + 1, rr2[n2], ccc, 0]
                            iz2 = self.boxes[bb + 1, rr2[n2], ccc, 1]
                            if ix == ix2 and iz == iz2:
                                cc2[n2] = ccc

                    n2limit = int(min(self.present[bb, iz, ix], numrays2))
                    omega1 = omega
                    omega2 = omega
                    for n2 in range(n2limit):
                        ne = self.eden[iz, ix]
                        epsilon = 1.0 - ne / ncrit
                        kmag = (omega / c) * np.sqrt(epsilon)

                        kx1 = kmag * (dkx[bb, rr1, cc1] / dkmag[bb, rr1, cc1] + 1.0e-10)
                        kx2 = kmag * (dkx[bb + 1, rr2[n2], cc2[n2]] / (dkmag[bb + 1, rr2[n2], cc2[n2]] + 1.0e-10))

                        kz1 = kmag * (dkz[bb, rr1, cc1] / (dkmag[bb, rr1, cc1] + 1.0e-10))
                        kz2 = kmag * (dkz[bb + 1, rr2[n2], cc2[n2]] / (dkmag[bb + 1, rr2[n2], cc2[n2]] + 1.0e-10))

                        kiaw = np.sqrt((kx2 - kx1) ** 2 + (kz2 - kz1) ** 2)
                        ws = kiaw * cs

                        eta = ((omega2 - omega1) - (kx2 - kx1) * self.u_flow[iz, ix]) / (ws + 1.0e-10)

                        efield1 = np.sqrt(8.0 * np.pi * 1.0e7 * i_b1[iz, ix] / c)

                        P = (iaw ** 2 * eta) / ((eta ** 2 - 1.0) ** 2 + iaw ** 2 * eta ** 2)
                        gain2 = constant1 * efield1 ** 2 * (ne / ncrit) * (1 / iaw) * P

                        if dkmag[bb + 1, rr2[n2], cc2[n2]] >= 1.0 * dx:
                            W2_new[iz, ix] = W2[iz, ix] * np.exp(
                                -1 * W1[iz, ix] * dkmag[bb + 1, rr2[n2], cc2[n2]] * gain2 / np.sqrt(epsilon))
                            W1_new[iz, ix] = W1[iz, ix] * np.exp(
                                1 * W2[iz, ix] * dkmag[bb, rr1, cc1] * gain2 / np.sqrt(epsilon))

    def calculate_intensity(self, bb, W1, W2, W1_new, W2_new, i_b1, i_b2, i_b1_new, i_b2_new):
        for rr1 in range(nrays):
            for cc1 in range(ncrossings):
                if self.boxes[bb, rr1, cc1, 0] == 0 or self.boxes[bb, rr1, cc1, 1] == 0:
                    break

                ix = self.boxes[bb, rr1, cc1, 0]
                iz = self.boxes[bb, rr1, cc1, 1]

                if self.intersections[iz, ix] != 0:
                    nonzeros1 = self.marked[bb, :, iz, ix].nonzero()
                    numrays1 = np.count_nonzero(self.marked[bb, :, iz, ix])

                    nonzeros2 = self.marked[bb + 1, :, iz, ix].nonzero()
                    numrays2 = np.count_nonzero(self.marked[bb + 1, :, iz, ix])

                    marker1 = self.marked[bb, nonzeros1, iz, ix].flatten()
                    marker2 = self.marked[bb + 1, nonzeros2, iz, ix].flatten()

                    rr2 = marker2
                    cc2 = marker2

                    for rrr in range(numrays1):
                        if marker1[rrr] == rr1:
                            ray1num = rrr
                            break

                    for n2 in range(numrays2):
                        for ccc in range(ncrossings):
                            ix2 = self.boxes[bb + 1, rr2[n2], ccc, 0]
                            iz2 = self.boxes[bb + 1, rr2[n2], ccc, 1]
                            if ix == ix2 and iz == iz2:
                                cc2[n2] = ccc
                                break

                    frac_change_1 = -1.0 * (1.0 - (W1_new[iz, ix] / W1[iz, ix])) * i_b1[iz, ix]
                    frac_change_2 = -1.0 * (1.0 - (W2_new[iz, ix] / W2[iz, ix])) * i_b2[iz, ix]

                    i_b1_new[iz, ix] += frac_change_1
                    i_b2_new[iz, ix] += frac_change_2

                    x_prev_1 = self.x[iz, ix]
                    z_prev_1 = self.z[iz, ix]

                    x_prev_2 = self.x[iz, ix]
                    z_prev_2 = self.z[iz, iz]

                    for ccc in range(cc1 + 1, ncrossings):
                        ix_next_1 = self.boxes[bb, rr1, ccc, 0]
                        iz_next_1 = self.boxes[bb, rr1, ccc, 1]

                        x_curr_1 = self.x[iz_next_1, ix_next_1]
                        z_curr_1 = self.z[iz_next_1, ix_next_1]

                        if ix_next_1 == 0 or iz_next_1 == 0:
                            break
                        else:
                            if x_curr_1 != x_prev_1 or z_curr_1 != z_prev_1:
                                i_b1_new[iz_next_1, ix_next_1] += frac_change_1 * (
                                        self.present[bb, iz, ix] / self.present[bb, iz_next_1, ix_next_1])
                            x_prev_1 = x_curr_1
                            z_prev_1 = z_curr_1

                    n2 = min(ray1num, numrays2)

                    for ccc in range(cc2[n2] + 1, ncrossings):
                        ix_next_2 = self.boxes[bb + 1, rr2[n2], ccc, 0]
                        iz_next_2 = self.boxes[bb + 1, rr2[n2], ccc, 1]

                        x_curr_2 = self.x[iz_next_2, ix_next_2]
                        z_curr_2 = self.z[iz_next_2, ix_next_2]

                        if ix_next_2 == 0 or iz_next_2 == 0:
                            break
                        else:
                            if x_curr_2 != x_prev_2 or z_curr_2 != z_prev_2:
                                i_b2_new[iz_next_2, ix_next_2] += frac_change_2 * (
                                            self.present[bb, iz, ix] / self.present[bb + 1, iz_next_2, ix_next_2])
                            x_prev_2 = x_curr_2
                            z_prev_2 = z_curr_2