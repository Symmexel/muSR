import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, h, mu_0
from matplotlib import pyplot as plt
from numpy.random import default_rng
from itertools import product
import gc

from scipy.linalg import logm

from quantum import Op, reduce_trace

# from numpy.fft import rfft

# from cProfile import Profile
# import pstats
# pstats.Stats(pr).sort_stats('cumulative').print_stats()

def adj(a):
    return a.conj().T

def tp(*args):
    result = args[0]
    for t in args[1:]:
        result = np.kron(result, t)
    return result

FLUORINE_19_GYROMAGNETIC_RATIO = 251.662e6
MUON_GYROMAGNETIC_RATIO_OVER_2_PI = 135.5e6
MUON_GYROMAGNETIC_RATIO = MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi
print(MUON_GYROMAGNETIC_RATIO)


I = np.eye(2)
sigma = np.array([
    [
        [0, 1],
        [1, 0]
    ],
    [
        [0, -1j],
        [1j, 0]
    ],
    [
        [1, 0],
        [0, -1]
    ]
])
S = (hbar/2) * sigma

rng = default_rng()

class Hamiltonian:
    def __init__(self):
        self.particle_count = 1
        self.fluorines = {}
        self.interactions = set()
    def add_fluorine(self, position):
        i = self.particle_count
        self.particle_count += 1
        self.fluorines[i] = position
        return i
    def add_interaction(self, i, j):
        if i == j:
            return
        elif j < i:
            i, j = j, i
        self.interactions.add((i, j))
    def add_nn_interactions(self, i, nn_radius):
        if i == 0:
            a = np.array([0, 0, 0])
        else:
            a = self.fluorines[i]
            if norm(a) <= nn_radius:
                self.add_interaction(0, i)
        for j in range(1, self.particle_count):
            b = self.fluorines[j]
            if norm(b - a) <= nn_radius:
                self.add_interaction(i, j)
    def add_all_interactions(self, i):
        self.add_interaction(0, i)
        for j in range(1, self.particle_count):
            self.add_interaction(i, j)
    def add(self, positions, nn_radius=None):
        if nn_radius is None:
            for r in positions:
                self.add_all_interactions(self.add_fluorine(r))
        else:
            for r in positions:
                self.add_nn_interactions(self.add_fluorine(r), nn_radius)
        return self
    def build(self):
        H = np.zeros((2**self.particle_count, 2**self.particle_count), dtype='complex128')
        for (i, j) in self.interactions:
            if i == 0:
                a = np.array([0, 0, 0])
            else:
                a = self.fluorines[i]
            b = self.fluorines[j]
            r = b - a
            d = norm(r)
            n = r / d
            Sn = np.tensordot(S, n, axes=(0, 0))
            if i == 0:
                H += (
                    mu_0 *
                    FLUORINE_19_GYROMAGNETIC_RATIO *
                    MUON_GYROMAGNETIC_RATIO_OVER_2_PI / (2*d**3)
                ) * (
                    sum(tp(S[k], *[I]*(j-1), S[k], *[I]*(self.particle_count-j-1)) for k in range(3))
                    - 3*tp(Sn, *[I]*(j-1), Sn, *[I]*(self.particle_count-j-1))
                )
            else:
                H += (
                    mu_0 *
                    FLUORINE_19_GYROMAGNETIC_RATIO *
                    FLUORINE_19_GYROMAGNETIC_RATIO / (4*np.pi*d**3)
                ) * (
                    sum(tp(*[I]*i, S[k], *[I]*(j-i-1), S[k], *[I]*(self.particle_count-j-1)) for k in range(3))
                    - 3*tp(*[I]*i, Sn, *[I]*(j-i-1), Sn, *[I]*(self.particle_count-j-1))
                )
        return H
    def second_moment(self, nn_radius, scaling=1):
        return (2/3) * (mu_0/(4*np.pi))**2 * hbar**2 * MUON_GYROMAGNETIC_RATIO**2 * FLUORINE_19_GYROMAGNETIC_RATIO**2 * (3/4) * (
            sum(pow(norm(r), -6) for r in self.fluorines.values() if norm(r) <= nn_radius)
            + sum(pow(scaling*norm(r), -6) for r in self.fluorines.values() if norm(r) > nn_radius)
        )

class FixedModelWithB:
    def __init__(self, H, beam_axis, measurement_axis, B=0, B_axis=np.array([0, 0, 1])):
        N = int(np.log2(H.shape[0]))
        self.energies, self.M = eigh(H + zeeman_term(N, B, B_axis))
        self.transition_amplitude_matrix = (
            np.abs(adj(self.M) @ Op(sigma).dot(beam_axis)((0, N)) @ self.M)
            * np.abs(adj(self.M) @ Op(sigma).dot(measurement_axis)((0, N)) @ self.M)
        ) / H.shape[0]
        self.rounded_frequencies = np.round(self.energies / h)
    def print_details(self):
        # J = S
        # S_muon = np.array([tp(s, I) for s in S])
        # S_fluorine = np.array([tp(I, s) for s in S])
        # J = (S_muon + S_fluorine) / hbar
        S_muon = np.array([tp(s, I, I) for s in S])
        S_fluorine1 = np.array([tp(I, s, I) for s in S])
        S_fluorine2 = np.array([tp(I, I, s) for s in S])
        J = (S_muon + S_fluorine1 + S_fluorine2) / hbar
        J2 = sum(J[i] @ J[i] for i in range(3))
        Jz = J[2]

        up_arrow = '\u2191'
        down_arrow = '\u2193'
        N = len(self.energies)
        particle_count = int(np.log2(len(self.energies)))
        for i in range(N):
            for j in range(particle_count-1, -1, -1):
                if i & (1 << j) == 0:
                    print(up_arrow, end='')
                else:
                    print(down_arrow, end='')
            print('|', end='')
        print()
        for i in range(N):
            for j in range(N):
                entry = self.M[j, i]
                if entry > 1e-8:
                    print('+'+' '*(particle_count-1), end='|')
                elif entry < -1e-8:
                    print('-'+' '*(particle_count-1), end='|')
                else:
                    print(' '*particle_count, end='|')
            print(
                f'j(j+1)={(adj(self.M[:, i]) @ J2 @ self.M[:, i]).real.round(2)}'
                f', m={(adj(self.M[:, i]) @ Jz @ self.M[:, i]).real.round(2)}'
                f', E={self.rounded_frequencies[i]} Hz'
            )
    def polarisation(self, ts):
        N = len(self.energies)
        frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / hbar
        return sum(
            self.transition_amplitude_matrix[i, j] * np.cos(frequency_matrix[i, j]*ts)
            for i in range(N) for j in range(N)
        )
    def plot_cosine_amplitudes(self, ax, colour=None):
        N = len(self.energies)
        frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / h

        cos_amplitudes = {}

        for i in range(N):
            for j in range(N):
                cos_amplitudes[frequency_matrix[i, j].round()] = \
                    cos_amplitudes.get(frequency_matrix[i, j].round(), 0) + self.transition_amplitude_matrix[i, j]

        for (f, a) in cos_amplitudes.items():
            ax.axvline(f * 1e-6, 0, a, color=colour)

        ax.set_title('Frequency Domain')
        ax.set_xlabel('Frequency / $MHz$')
        ax.set_ylabel('Amplitude')
    def plot_energy_levels(self, ax, colour=None):
        N = len(self.energies)
        rounded_frequencies = self.energies / h#np.round(self.energies / h)
        frequency_levels = set(rounded_frequencies)
        transitions = set()
        for i in range(N):
            for j in range(N):
                if self.transition_amplitude_matrix[i, j] > 1e-4:#1e-8 or True:
                # if self.transition_amplitude_matrix[i, j] > 1e-8 or True:
                    a = rounded_frequencies[i]
                    b = rounded_frequencies[j]
                    if a > b:
                        a, b = b, a
                    if a != b:
                        # if (a, b) not in transitions:
                        print(a, b, b-a, self.transition_amplitude_matrix[i, j])#
                        transitions.add((a, b))
                if self.transition_amplitude_matrix[i, j] < 0:
                    a = rounded_frequencies[i]
                    b = rounded_frequencies[j]
                    if a > b:
                        a, b = b, a
                    if a != b:
                        # if (a, b) not in transitions:
                        print(a, b, b-a, self.transition_amplitude_matrix[i, j])#
                        transitions.add((a, b))
        for f in frequency_levels:
            ax.axhline(f/1e6, 0, len(transitions)+1, color=colour)
        x = 1
        for (a, b) in transitions:
            ax.annotate('', (x, a/1e6), (x, b/1e6), arrowprops={'arrowstyle': '<->'})
            x += 1
        ax.set_title('Muon spin energy levels')
        ax.set_xlim(0, len(transitions)+1)
        ax.set_ylabel('Frequency / $MHz$')
        ax.get_xaxis().set_visible(False)

def zeeman_term(particle_count, B, B_axis=np.array([0, 0, 1])):
    Sn = np.tensordot(S, B_axis, axes=(0, 0))
    return -B*(
        MUON_GYROMAGNETIC_RATIO*tp(Sn, *[I]*(particle_count-1))
        + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), Sn, *[I]*(particle_count-j-1)) for j in range(1, particle_count))
    )

class SolveH:
    def __init__(
            self,
            H,
            x_axis=np.array([1, 0, 0]),
            y_axis=np.array([0, 1, 0]),
            z_axis=np.array([0, 0, 1]),
            beam_axis=np.array([0, 0, 1]),
            measurement_axis=np.array([0, 0, 1]),
            attenuation=1
        ):
        self.H = H
        self.beam_axis = beam_axis
        self.measurement_axis = measurement_axis
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.attenuation = attenuation
        self.particle_count = int(np.log2(self.H.shape[0]))
    def polarisation(self, ts, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_beam = tp(np.tensordot(sigma, self.beam_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        sigma_measure = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_beam @ adj(U) @ sigma_measure).real
        for i in range(1, len(ts)):
            energies, M = eigh(H+H1(ts[i-1], self))
            U = M @ np.diag(np.exp(energies*(-1j*(ts[i]-ts[i-1])/hbar))) @ adj(M) @ U
            D[i] = np.trace(U @ sigma_beam @ adj(U) @ sigma_measure).real
        D /= N
        return D
    def polarisation_optimised(self, ts, period, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_beam = tp(np.tensordot(sigma, self.beam_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        sigma_measure = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_beam @ adj(U) @ sigma_measure).real
        dt = ts[1] - ts[0]
        dU = np.ndarray((int(np.ceil(period / dt)), *U.shape), np.complex)
        for i in range(dU.shape[0]):
            energies, M = eigh(H+H1(ts[i], self))
            dU[i] = M @ np.diag(np.exp(energies*(-1j*dt/hbar))) @ adj(M)
        for i in range(1, len(ts)):
            U = dU[int((ts[i]%period)//dt)] @ U
            D[i] = np.trace(U @ sigma_beam @ adj(U) @ sigma_measure).real
        D /= N
        return D
    def entanglement(self, ts, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        dim = self.H.shape[0]
        N = int(np.log2(dim))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        density_matrix = (np.eye(dim) + sigma_mu)/dim
        E = np.zeros((N, ts.shape[0]))
        for j in range(N):
            reduced_density_matrix = reduce_trace(density_matrix, j)
            E[j, 0] = np.trace(reduced_density_matrix @ reduced_density_matrix).real
            # E[j, 0] = np.trace(reduced_density_matrix @ np.tensordot(sigma, self.measurement_axis, axes=(0, 0))).real
        for i in range(1, len(ts)):
            energies, M = eigh(H+H1(ts[i-1], self))
            dU = M @ np.diag(np.exp(energies*(-1j*(ts[i]-ts[i-1])/hbar))) @ adj(M)
            density_matrix = dU @ density_matrix @ adj(dU)
            for j in range(N):
                reduced_density_matrix = reduce_trace(density_matrix, j)
                E[j, i] = np.trace(reduced_density_matrix @ reduced_density_matrix).real
                # E[j, i] = np.trace(reduced_density_matrix @ np.tensordot(sigma, self.measurement_axis, axes=(0, 0))).real
        # print(reduce_trace(density_matrix, j))
        return E
    def entanglement_optimised(self, ts, period, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        dim = self.H.shape[0]
        N = int(np.log2(dim))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(dim)
        initial_density_matrix = (np.eye(dim) + sigma_mu)/dim
        E = np.zeros((N, ts.shape[0]))
        for j in range(N):
            reduced_density_matrix = reduce_trace(initial_density_matrix, j)
            E[j, 0] = np.trace(reduced_density_matrix @ reduced_density_matrix).real
        dt = ts[1] - ts[0]
        dU = np.ndarray((int(np.ceil(period / dt)), *U.shape), np.complex)
        for i in range(dU.shape[0]):
            energies, M = eigh(H+H1(ts[i], self))
            dU[i] = M @ np.diag(np.exp(energies*(-1j*dt/hbar))) @ adj(M)
        for i in range(1, len(ts)):
            U = dU[int((ts[i]%period)//dt)] @ U
            density_matrix = U @ initial_density_matrix @ adj(U)
            for j in range(N):
                reduced_density_matrix = reduce_trace(density_matrix, j)
                E[j, i] = np.trace(reduced_density_matrix @ reduced_density_matrix).real
        self.energies, M = eigh(H+H1(ts[-2], self))
        self.transition_amplitude_matrix = (
            np.abs(adj(M) @ U @ sigma_mu @ adj(U) @ M)
            * np.abs(adj(M) @ sigma_mu @ M)
        ) / N
        return E
    def calculate_transition_amplitude_matrix(self, ts, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_beam = tp(np.tensordot(sigma, self.beam_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        sigma_measure = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        M = None
        for i in range(1, len(ts)):
            energies, M = eigh(H+H1(ts[i-1], self))
            U = M @ np.diag(np.exp(energies*(-1j*(ts[i]-ts[i-1])/hbar))) @ adj(M) @ U
        assert M is not None
        energies, M = eigh(H)
        self.energies = energies#
        self.M = M#
        self.rounded_frequencies = np.round(self.energies / h)#
        
        # # S_muon = np.array([tp(s, I) for s in S])
        # # S_fluorine = np.array([tp(I, s) for s in S])
        # # J = (S_muon + S_fluorine) / hbar
        # S_muon = np.array([tp(s, I, I) for s in S])
        # S_fluorine1 = np.array([tp(I, s, I) for s in S])
        # S_fluorine2 = np.array([tp(I, I, s) for s in S])
        # J = (S_muon + S_fluorine1 + S_fluorine2) / hbar
        # J2 = sum(J[i] @ J[i] for i in range(3))
        # # density_matrix = (np.eye(2**self.particle_count) + sigma_beam)/4
        # density_matrix = tp(np.array([[1, 0], [0, 0]]), *[I]*(self.particle_count-1)) / (self.particle_count-1)**2

        # for k in range(len(energies)):
        #     if adj(M[:, k]) @ J2 @ M[:, k] > 1.7:
        #         n = np.linalg.lstsq(np.tensordot(J, M[:, k], 1).T, M[:, k], rcond=None)[0].real
        #         theta = np.arccos(n[2])
        #         phi = np.arctan(n[1]/n[0]) if 0 < theta < np.pi else 0
        #         print(f'theta={theta}, phi={phi}')
        #         Ja = np.tensordot(J, n, (0, 0))
        # for k in range(len(energies)):
        #     if np.abs(np.round(energies[k]/h)) == 10724825 or True:
        #         print(energies[k]/h)
        #         print(M[:, k])
                
        #         print(f'j(j+1)={adj(M[:, k]) @ J2 @ M[:, k]}')
        #         print(f'm={adj(M[:, k]) @ Ja @ M[:, k]}')

        #         print(adj(M[:, k]) @ sigma_beam @ M[:, k])
        #         # print(np.trace(np.outer(M[:, k], M[:, k].conj()) @ sigma_beam))
        #         print(adj(M[:, k]) @ U @ sigma_beam @ adj(U) @ M[:, k])
        #         # print(np.trace(np.outer(M[:, k], M[:, k].conj()) @ U @ sigma_beam @ adj(U)))
        #         print(adj(M[:, k]) @ density_matrix @ M[:, k])
        #         print(adj(M[:, k]) @ U @ density_matrix @ adj(U) @ M[:, k])
        #         print(adj(M[:, k]) @ sigma_measure @ M[:, k])
        #         # print(np.trace(np.outer(M[:, k], M[:, k].conj()) @ sigma_measure))
        #         print(adj(M[:, k]) @ U @ sigma_measure @ adj(U) @ M[:, k])
        #         # print(np.trace(np.outer(M[:, k], M[:, k].conj()) @ U @ sigma_measure @ adj(U)))
        # print('>', (adj(M) @ U @ M).round(2))#M
        # print(sigma_beam.round(2))
        # print((U @ sigma_beam @ adj(U)).round(2))
        # print(reduce_trace(U @ sigma_beam @ adj(U), 0))
        # print(reduce_trace(U @ sigma_beam @ adj(U), 1))
        # print('>', (adj(M) @ U @ density_matrix @ adj(U) @ M).round(2))#M
        # # def random_row():
        # #     state = rng.random(2) + 1j*rng.random(2)
        # #     state /= norm(state)
        # #     return np.outer(state, adj(state))
        # # density_matrix = tp(np.array([[1, 0], [0, 0]]), *(random_row() for i in range(1, self.particle_count)))
        # # row = reduce_trace(U @ density_matrix @ adj(U), 0)
        # row = reduce_trace(U @ density_matrix @ adj(U), 0)
        # print(row.round(2))
        # print(eigh(row))
        # print(np.trace(np.linalg.matrix_power(row, 2)))
        # print(-np.trace(row @ logm(row)), np.arccos(self.beam_axis[2]))
        
        # return (
        #     np.abs(adj(M) @ U @ sigma_beam @ adj(U) @ M)
        #     * np.abs(adj(M) @ sigma_measure @ M)
        # ) / N
        self.transition_amplitude_matrix = (
            np.abs(adj(M) @ U @ sigma_beam @ adj(U) @ M)
            * np.abs(adj(M) @ sigma_measure @ M)
        ) / N
    def calculate_transition_amplitude_matrix_periodic(self, ts, period, H1=lambda t, model: 0, H1_prep=None):
        plt.figure()
        ax = plt.gca(projection='3d')

        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_beam = tp(np.tensordot(sigma, self.beam_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        sigma_measure = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        dt = ts[1] - ts[0]
        dU = np.ndarray((int(np.ceil(period / dt)), *U.shape), np.complex)
        for i in range(dU.shape[0]):
            energies, M = eigh(H+H1(ts[i], self))
            dU[i] = M @ np.diag(np.exp(energies*(-1j*dt/hbar))) @ adj(M)
        xs = np.ndarray(ts.shape, dtype=float)
        ys = np.ndarray(ts.shape, dtype=float)
        zs = np.ndarray(ts.shape, dtype=float)
        row = reduce_trace(U @ sigma_beam @ adj(U), 0)
        xs[0] = np.trace(reduce_trace(self.matrix_x, 0) @ row).real
        ys[0] = np.trace(reduce_trace(self.matrix_y, 0) @ row).real
        zs[0] = np.trace(reduce_trace(self.matrix_z, 0) @ row).real
        for i in range(1, len(ts)):
            U = dU[int((ts[i]%period)//dt)] @ U
            row = reduce_trace(U @ sigma_beam @ adj(U), 0)
            xs[i] = np.trace(reduce_trace(self.matrix_x, 0) @ row).real
            ys[i] = np.trace(reduce_trace(self.matrix_y, 0) @ row).real
            zs[i] = np.trace(reduce_trace(self.matrix_z, 0) @ row).real
        ax.plot(xs, ys, zs)
        plt.figure()
        self.energies, M = eigh(H+H1(ts[-2], self))
        self.M = M#
        self.rounded_frequencies = np.round(self.energies / h)#
        self.transition_amplitude_matrix = (
            np.abs(adj(M) @ U @ sigma_beam @ adj(U) @ M)
            * np.abs(adj(M) @ sigma_measure @ M)
        ) / N
        gc.collect()
    def polarisation_continue(self, ts):
        dim = len(self.energies)
        frequency_matrix = np.abs(np.tile(self.energies, (dim, 1)).T - np.tile(self.energies, (dim, 1))) / hbar
        return sum(
            self.transition_amplitude_matrix[i, j] * np.cos(frequency_matrix[i, j]*ts)
            for i in range(dim) for j in range(dim)
        )
    def find_frequencies(self, cos_amplitudes, factor, decimals=0):
        N = len(self.energies)
        # frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / h

        for i in range(N):
            for j in range(N):
                # cos_amplitudes[frequency_matrix[i, j].round()] = \
                #     cos_amplitudes.get(frequency_matrix[i, j].round(), 0) + factor*self.transition_amplitude_matrix[i, j]
                f = (self.energies[i] - self.energies[j]) / h
                cos_amplitudes[f.round(decimals)] = \
                    cos_amplitudes.get(f.round(decimals), 0) + factor*self.transition_amplitude_matrix[i, j]
    def add_spectra(self, bins, bin_start, bin_end, bin_width):
        dim = len(self.energies)
        for i in range(dim):
            for j in range(dim):
                f = (self.energies[i] - self.energies[j]) / h
                if bin_start <= f <= bin_end:
                    bins[int((f - bin_start) / bin_width)] += self.transition_amplitude_matrix[i, j]

class ForwardEuler:
    def __init__(self, H, measurement_axis=np.array([0, 0, 1]), x_axis=np.array([1, 0, 0]), y_axis=np.array([0, 1, 0])):
        self.H = H
        self.measurement_axis = measurement_axis
        self.x_axis = x_axis
        self.y_axis = y_axis
    def polarisation(self, ts):
        N = self.H.shape[0]
        particle_count = int(np.log2(N))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        for i in range(1, len(ts)):
            U = U - (1j * (ts[i] - ts[i-1]) / hbar) * H @ U
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D

class CN:
    def __init__(self, H, measurement_axis=np.array([0, 0, 1]), x_axis=np.array([1, 0, 0]), y_axis=np.array([0, 1, 0])):
        self.H = H
        self.measurement_axis = measurement_axis
        self.x_axis = x_axis
        self.y_axis = y_axis
    def polarisation(self, ts, H1=lambda t, model: 0):
        N = self.H.shape[0]
        particle_count = int(np.log2(N))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        for i in range(1, len(ts)):
            U = np.linalg.solve(
                np.eye(N) + (1j * (ts[i] - ts[i-1]) / 2 / hbar)*(H+H1(ts[i], self)),
                (np.eye(N) - (1j * (ts[i] - ts[i-1]) / 2 / hbar)*(H+H1(ts[i-1], self))) @ U
            )
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D

class Heuns:
    def __init__(self, H, measurement_axis=np.array([0, 0, 1]), x_axis=np.array([1, 0, 0]), y_axis=np.array([0, 1, 0])):
        self.H = H
        self.measurement_axis = measurement_axis
        self.x_axis = x_axis
        self.y_axis = y_axis
    def polarisation(self, ts, H1=lambda t, model: 0):
        N = self.H.shape[0]
        particle_count = int(np.log2(N))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        for i in range(1, len(ts)):
            U1 = U - (1j * (ts[i] - ts[i-1]) / hbar) * (H+H1(ts[i-1], self)) @ U
            U2 = U - (1j * (ts[i] - ts[i-1]) / hbar) * (H+H1(ts[i], self)) @ U1
            U = (U1 + U2) / 2
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D

class ModelWithB:
    def __init__(self, H, B=0, sample_count=10):
        bin_size = sample_count
        thetas = np.arccos(1 - 2*np.array([rng.random(sample_count//bin_size)/bin_size + i/bin_size for i in range(bin_size)]).flatten())
        phis = 2*np.pi*rng.random(sample_count)
        self.samples = []
        for i in range(sample_count):
            B_axis = np.array([
                np.sin(thetas[i])*np.cos(phis[i]),
                np.sin(thetas[i])*np.sin(phis[i]),
                np.cos(thetas[i])
            ])
            self.samples.append(FixedModelWithB(H, B, B_axis))
    def polarisation(self, ts):
        return sum(m.polarisation(ts) for m in self.samples) / len(self.samples)

class TimeDepModelWithB:
    def __init__(self, H, B=0, sample_count=10, fixed_model=SolveH):
        particle_count = int(np.log2(H.shape[0]))
        bin_size = sample_count
        thetas = np.arccos(1 - 2*np.array([rng.random(sample_count//bin_size)/bin_size + i/bin_size for i in range(bin_size)]).flatten())
        phis = 2*np.pi*rng.random(sample_count)
        psis = 2*np.pi*rng.random(sample_count)
        self.thetas = thetas
        self.samples = []
        for i in range(sample_count):
            # thetas[i] = np.pi/3
            # phis[i] = 0
            x_axis = np.array([
                np.cos(thetas[i])*np.cos(phis[i]),
                np.cos(thetas[i])*np.sin(phis[i]),
                -np.sin(thetas[i])
            ])
            y_axis = np.array([
                -np.sin(phis[i]),
                np.cos(phis[i]),
                0
            ])
            z_axis = np.array([
                np.sin(thetas[i])*np.cos(phis[i]),
                np.sin(thetas[i])*np.sin(phis[i]),
                np.cos(thetas[i])
            ])

            # Apply third rotation!
            # Tom, I only remembered the third rotation while writing the method part of the report
            R1 = np.array([x_axis, y_axis, z_axis]).T
            R2 = np.array([
                [
                    np.cos(psis[i]) + z_axis[0]**2*(1-np.cos(psis[i])),
                    z_axis[0]*z_axis[1]*(1-np.cos(psis[i])) - z_axis[2]*np.sin(psis[i]),
                    z_axis[2]*z_axis[0]*(1-np.cos(psis[i])) + z_axis[1]*np.sin(psis[i])
                ],
                [
                    z_axis[0]*z_axis[1]*(1-np.cos(psis[i])) + z_axis[2]*np.sin(psis[i]),
                    np.cos(psis[i]) + z_axis[1]**2*(1-np.cos(psis[i])),
                    z_axis[1]*z_axis[2]*(1-np.cos(psis[i])) - z_axis[0]*np.sin(psis[i]),
                ],
                [
                    z_axis[2]*z_axis[0]*(1-np.cos(psis[i])) - z_axis[1]*np.sin(psis[i]),
                    z_axis[1]*z_axis[2]*(1-np.cos(psis[i])) + z_axis[0]*np.sin(psis[i]),
                    np.cos(psis[i]) + z_axis[2]**2*(1-np.cos(psis[i])),
                ]
            ])
            R_total = R2 @ R1
            x_axis = R_total[:, 0]
            y_axis = R_total[:, 1]
            z_axis = R_total[:, 2]

            attenuation = np.exp(-0.05*rng.random())
            beam_axis = z_axis
            measurement_axis = x_axis
            # self.samples.append(fixed_model(H + zeeman_term(particle_count, B, z_axis), x_axis, y_axis, z_axis, x_axis))

            # New!!!
            H, scaling = scale(central_unit_cell_positions[0:2], central_unit_cell_positions[2:], radial_unit_cell_positions, z_axis)
            if scaling < 0.98:
                print(f'theta={thetas[i]}, phi={phis[i]}, psi={psis[i]}: {scaling}')

            self.samples.append(fixed_model(H + attenuation*zeeman_term(particle_count, B, z_axis),
                x_axis, y_axis, z_axis, beam_axis, measurement_axis, attenuation))
            
            # FixedModelWithB(H, beam_axis, measurement_axis, B, z_axis).print_details()
            FixedModelWithB(H, beam_axis, measurement_axis, B, z_axis).plot_energy_levels(plt.gca())
            plt.figure()
    def polarisation(self, *args, **kwargs):
        return sum(m.polarisation(*args, **kwargs) for m in self.samples) / len(self.samples)
    def polarisation_optimised(self, *args, **kwargs):
        return sum(m.polarisation_optimised(*args, **kwargs) for m in self.samples) / len(self.samples)
    def entanglement(self, *args, **kwargs):
        return sum(m.entanglement(*args, **kwargs) for m in self.samples) / len(self.samples)
    def entanglement_optimised(self, *args, **kwargs):
        return sum(m.entanglement_optimised(*args, **kwargs) for m in self.samples) / len(self.samples)
    def calculate_transition_matrix_periodic(self, *args, **kwargs):
        for m in self.samples:
            m.calculate_transition_amplitude_matrix_periodic(*args, **kwargs)
        return self
    def polarisation_continue(self, *args, **kwargs):
        return sum(m.polarisation_continue(*args, **kwargs) for m in self.samples) / len(self.samples)
    def plot_cosine_amplitudes(self, ax, **kwargs):
        cos_amplitudes = {}
        for m in self.samples:
            m.find_frequencies(cos_amplitudes, 1/len(self.samples))

        data = []

        for (f, a) in cos_amplitudes.items():
            # if a > 1e-3:
            if f > 0:
                # ax.axvline(f / 1e6, 0, a, color=colour)
                # ax.plot([f / 1e6, f / 1e6], [0, a], color=colour, linestyle='-')
                data.append((f / 1e6, a))
        
        # # from matplotlib.collections import PatchCollection
        # # from matplotlib.patches import Rectangle
        # # pc = PatchCollection([Rectangle((f - 0.5, 0), 1, a) for (f, a) in data])
        # # ax.add_collection(pc)

        data_array = np.array(data)
        ax.vlines(data_array[:, 0], np.zeros(data_array.shape[0]), data_array[:, 1], **kwargs)

        # plt.tick_params(axis='y', which='both', left=False, labelleft=False)
        # ax.set_xlim((21.435, 21.475))
        # ax.set_ylim((0, 0.0012))
        ax.set_xlim((18, 25))
        ax.set_ylim(bottom=0)
        # ax.set_title('Frequency Domain')
        ax.set_xlabel(r'Frequency / $\mathrm{MHz}$')
        # ax.set_ylabel('Power')
    def plot_spectra(self, decimals, ax, **kwargs):
        cos_amplitudes = {}
        for m in self.samples:
            m.find_frequencies(cos_amplitudes, 1/len(self.samples), decimals)

        data = []

        for (f, a) in cos_amplitudes.items():
            # if a > 1e-3:
            if f > 0:
                # ax.axvline(f / 1e6, 0, a, color=colour)
                # ax.plot([f / 1e6, f / 1e6], [0, a], color=colour, linestyle='-')
                data.append((f / 1e6, a))
        
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        pc = PatchCollection([Rectangle((f - 0.5e-6*pow(10, -decimals), 0), 1e-6*pow(10, -decimals), a) for (f, a) in data])
        ax.add_collection(pc)

        # plt.tick_params(axis='y', which='both', left=False, labelleft=False)
        # ax.set_xlim((21.435, 21.475))
        # ax.set_ylim((0, 0.0012))
        ax.set_xlim((18, 25))
        ax.set_ylim(bottom=0)
        # ax.set_title('Frequency Domain')
        ax.set_xlabel(r'Frequency / $\mathrm{MHz}$')
        # ax.set_ylabel('Power')
    def calculate_spectra(self, bin_start, bin_end, bin_width, scale):
        self.bins = np.zeros(int(np.ceil(bin_end-bin_start) / bin_width))
        for m in self.samples:
            m.add_spectra(self.bins, bin_start, bin_end, bin_width)
        self.bins /= len(self.samples)
        xs = np.concatenate([
            (bin_start,),
            np.repeat(np.arange(bin_start+bin_width, bin_end, bin_width), 2),
            (bin_end,)
        ]) / scale
        return xs, np.repeat(self.bins, 2)
    def spectra_periodic(self, bin_start, bin_end, bin_width, scale, *args, **kwargs):
        bins = np.zeros(int(np.ceil(bin_end-bin_start) / bin_width))
        for i in range(len(self.samples)):
            self.samples[i].calculate_transition_amplitude_matrix_periodic(*args, **kwargs)
            # self.samples[i].calculate_transition_amplitude_matrix(*args, **kwargs)
            self.samples[i].add_spectra(bins, bin_start, bin_end, bin_width)
            # FixedModelWithB.print_details(self.samples[i])#
            FixedModelWithB.plot_energy_levels(self.samples[i], plt.gca())#
            plt.figure()#
            self.samples[i] = None
            gc.collect()
        bins /= len(self.samples)
        xs = np.concatenate([
            (bin_start,),
            np.repeat(np.arange(bin_start+bin_width, bin_end, bin_width), 2),
            (bin_end,)
        ]) / scale
        return xs, np.repeat(bins, 2)

B = 1583e-4
pulse_B = 18.4e-4
pulse_duration = 1e-6
pulse_angular_frequency = MUON_GYROMAGNETIC_RATIO*B
pulse2_B = FLUORINE_19_GYROMAGNETIC_RATIO*B

print(pulse_angular_frequency / (2*np.pi) / 1e6)
print(MUON_GYROMAGNETIC_RATIO)

MU_F_DISTANCE = 1.17e-10
a = 4.62e-10

H = Hamiltonian()
H.add_all_interactions(H.add_fluorine(np.array([0, 0, MU_F_DISTANCE])))
H.add_all_interactions(H.add_fluorine(np.array([0, 0, -MU_F_DISTANCE])))
# H.add_all_interactions(H.add_fluorine(np.array([MU_F_DISTANCE/np.sqrt(2), MU_F_DISTANCE/np.sqrt(2), 0])))
# H.add_all_interactions(H.add_fluorine(np.array([-MU_F_DISTANCE/np.sqrt(2), -MU_F_DISTANCE/np.sqrt(2), 0])))
# H_small_matrix = H.build()
H.add_all_interactions(H.add_fluorine(np.array([a/8, -a/8, a/4])))
H.add_all_interactions(H.add_fluorine(np.array([a/8, -a/8, -a/4])))
H.add_all_interactions(H.add_fluorine(np.array([-a/8, a/8, a/4])))
H.add_all_interactions(H.add_fluorine(np.array([-a/8, a/8, -a/4])))

central_unit_cell_positions = np.array([
    [MU_F_DISTANCE/np.sqrt(2), MU_F_DISTANCE/np.sqrt(2), 0],
    [-MU_F_DISTANCE/np.sqrt(2), -MU_F_DISTANCE/np.sqrt(2), 0],
    [a/8, -a/8, a/4],
    [a/8, -a/8, -a/4],
    [-a/8, a/8, a/4],
    [-a/8, a/8, -a/4]
])

radial_unit_cell_positions = np.array([
    [a/8, a/8, 0],
    [-a/8, -a/8, 0],
    [a/8, -a/8, a/4],
    [a/8, -a/8, -a/4],
    [-a/8, a/8, a/4],
    [-a/8, a/8, -a/4]
])

def second_moment_infinity(n):
    factor = (2/3) * (mu_0/(4*np.pi))**2 * hbar**2 * MUON_GYROMAGNETIC_RATIO**2 * FLUORINE_19_GYROMAGNETIC_RATIO**2 * (3/4)
    result = 0
    # result = sum(pow(norm(r), -6) for r in central_unit_cell_positions)
    for (i, j, k) in product(range(-n, n+1), repeat=3):
        if i == j == k == 0:
            result += sum(pow(norm(r), -6) for r in central_unit_cell_positions)
        else:
            result += sum(pow(norm(r), -6) for r in radial_unit_cell_positions + a*np.array([i, j, k]))
    return factor * result

from scipy.optimize import curve_fit

# New!!!
def scale(nn_positions, nnn_positions, radial_unit_cell_positions, q):
    def second_moment(_, scaling=1):
        return (
            sum(((1-3*(np.dot(r, q)/norm(r))**2)**2)*pow(norm(r), -6) for r in nn_positions)
            + sum(((1-3*(np.dot(scaling*r, q)/norm(scaling*r))**2)**2)*pow(scaling*norm(r), -6) for r in nnn_positions)
        )

    central_unit_cell_positions = np.concatenate((nn_positions, nnn_positions))
    def second_moment_infinity(n):
        result = 0
        # result = sum(pow(norm(r), -6) for r in central_unit_cell_positions)
        for (i, j, k) in product(range(-n, n+1), repeat=3):
            if i == j == k == 0:
                result += sum(((1-3*(np.dot(r, q)/norm(r))**2)**2)*pow(norm(r), -6) for r in central_unit_cell_positions)
            else:
                result += sum(((1-3*(np.dot(r, q)/norm(r))**2)**2)*pow(norm(r), -6) for r in radial_unit_cell_positions + a*np.array([i, j, k]))
        return result
    
    scaling = curve_fit(second_moment, None, np.array([second_moment_infinity(10)]), np.array([0.93]))[0][0]

    H_scaled = Hamiltonian()
    for r in nn_positions:
        H_scaled.add_all_interactions(H_scaled.add_fluorine(r))
    for r in nnn_positions:
        H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*r))
    return H_scaled.build(), scaling

# scaling = curve_fit(H.second_moment, MU_F_DISTANCE*1.1, np.array([second_moment_infinity(10)]), np.array([0.93]))[0][0]
# print(scaling)

# H_scaled = Hamiltonian()
# H_scaled.add_all_interactions(H_scaled.add_fluorine(np.array([MU_F_DISTANCE/np.sqrt(2), MU_F_DISTANCE/np.sqrt(2), 0])))
# H_scaled.add_all_interactions(H_scaled.add_fluorine(np.array([-MU_F_DISTANCE/np.sqrt(2), -MU_F_DISTANCE/np.sqrt(2), 0])))
# H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([a/8, -a/8, a/4])))
# H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([a/8, -a/8, -a/4])))
# H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([-a/8, a/8, a/4])))
# H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([-a/8, a/8, -a/4])))

def H_pulse_prep(model):
    model.matrix_x = model.attenuation*zeeman_term(model.particle_count, pulse_B, model.x_axis)
    model.matrix_y = model.attenuation*zeeman_term(model.particle_count, pulse_B, model.y_axis)
    model.matrix_z = model.attenuation*zeeman_term(model.particle_count, pulse_B, model.z_axis)

def H_pulse(t, model=None):
    return (
        np.cos(MUON_GYROMAGNETIC_RATIO*B*t) * model.matrix_x
        - np.sin(MUON_GYROMAGNETIC_RATIO*B*t) * model.matrix_y
    ) if t < pulse_duration else 0

# ts = np.arange(0, 1e-6, 1e-9)
# ts = np.arange(0, 1e-6, 1e-9) # <-
ts = np.arange(0, 10e-6, 1e-9)
# ts = np.arange(0, 5e-6, 1e-9)
# ts = np.arange(0, 10e-6, 1e-9)

# plt.title('Fourier Spectra for NaF Without Decoherence')
# m = TimeDepModelWithB(
#     H.build(), B, 20
# ).calculate_transition_matrix_periodic(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep
# )

# plt.title('Fourier Spectra for NaF with 100 samples')
# plt.plot(
#     ts,
#     TimeDepModelWithB(
#         H.build(), B, 100
#     ).polarisation_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep)
# )

# plt.title('Polarisation for NaF with 10 samples')
# plt.plot(
#     ts,
#     TimeDepModelWithB(
#         H.build(), B, 10
#     ).polarisation_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep)
# )
# plt.xlabel('Time / $s$')
# plt.ylabel('Muon spin polarisation')

# plt.title('Polarisation for NaF with 1 samples')
# plt.plot(
#     ts,
#     TimeDepModelWithB(
#         H.build(), B, 10
#     # ).polarisation_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep)
#     ).polarisation(ts, H_pulse, H_pulse_prep)
# )
# plt.xlabel('Time / $s$')
# plt.ylabel('Muon spin polarisation')


# Here!!!
plt.title('Fourier Spectra for FmuF with 1 samples')
plt.plot(
    *TimeDepModelWithB(
        H.build(), B, 1
    ).spectra_periodic(
        18e6, 24.5e6, 50, 1e6,
        ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep
        # ts, H_pulse, H_pulse_prep
    )
)
plt.xlabel(r'Frequency / $\mathrm{MHz}$')
plt.gca().axes.yaxis.set_visible(False)
plt.ylim(bottom=0)

# TimeDepModelWithB(
#     H.build(), B, 3
# ).calculate_transition_matrix_periodic(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep
# ).plot_cosine_amplitudes(plt.gca())

# plt.title('Fourier Spectra for NaF with 1000 samples')
# plt.plot(
#     *TimeDepModelWithB(
#         H.build(), B, 1000
#     ).spectra_periodic(
#         # 20.5e6, 22.5e6, 50, 1e6,
#         18e6, 24.5e6, 50, 1e6,
#         # 18e6, 22.5e6, 1e3, 1e6,
#         ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep
#     )
# )
# plt.xlabel(r'Frequency / $\mathrm{MHz}$')
# plt.gca().axes.yaxis.set_visible(False)

# plt.figure()

# plt.title('Fourier Spectra for NaF with 10,000 samples')
# plt.plot(
#     *TimeDepModelWithB(
#         H.build(), B, 10_000
#     ).spectra_periodic(
#         # 20.5e6, 22.5e6, 50, 1e6,
#         18e6, 24.5e6, 50, 1e6,
#         # 18e6, 22.5e6, 1e3, 1e6,
#         ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep
#     )
# )
# plt.xlabel(r'Frequency / $\mathrm{MHz}$')
# plt.gca().axes.yaxis.set_visible(False)

# plt.show()

# plt.title('Fourier Spectra for NaF with bin width 1 Hz')
# m.plot_cosine_amplitudes(plt.gca(), color='blue')
# plt.figure()
# plt.title('Fourier Spectra for NaF with bin width 1 kHz')
# m.plot_spectra(-3, plt.gca(), color='blue')
# ).plot_cosine_amplitudes(plt.gca(), color='blue')
# ).plot_spectra(-3, plt.gca(), color='blue')

# plt.title('Fourier Spectra for NaF')
# plt.plot(*m.spectra_periodic(18e6, 25e6, 1e1, 1e6, ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep))
# # plt.plot(*m.calculate_spectra(18e6, 25e6, 1e1, 1e6))
# plt.xlabel(r'Frequency / $\mathrm{MHz}$')

# plt.figure()
# plt.title('Fourier Spectra for NaF With Decoherence')
# TimeDepModelWithB(
#     H_scaled.build(), B, 20
# ).calculate_transition_matrix_periodic(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep
# ).plot_cosine_amplitudes(plt.gca(), color='orange')
# plt.plot(
#     ts,
#     TimeDepModelWithB(
#         H_small_matrix, B, 20
#     ).polarisation_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep),
#     label=r'$\mathrm{F{\mu}F}$'
# )

# m = TimeDepModelWithB(H.build(), B, 10)
# # e = m.entanglement_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep)
# e = m.entanglement(ts, H_pulse, H_pulse_prep)
# for j in range(e.shape[0]):
#     plt.plot(ts, e[j], label=r'$\mathrm{F{\mu}F+NNN}$ 'f'{j}')
# m_small = TimeDepModelWithB(H_small_matrix, B, 10)
# e = m_small.entanglement_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep)
# e = m_small.entanglement(ts, H_pulse, H_pulse_prep)
# for j in range(e.shape[0]):
#     plt.plot(ts, e[j], label=r'$\mathrm{F{\mu}F}$ 'f'{j}', linestyle='dotted')

# plt.plot(
#     ts,
#     TimeDepModelWithB(
#         H.build(), B, 20
#     ).polarisation_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep),
#     label=r'$\mathrm{F{\mu}F+NNN}$'
# )
# plt.legend()
# plt.yscale('log')

# plt.title(r'Powder average, $\mathrm{NaF+\mu}$')
# plt.plot(ts, TimeDepModelWithB(H_small_matrix, B, 20)
#     .polarisation_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep))
# plt.plot(ts, TimeDepModelWithB(H.build(), B, 20)
#     .polarisation_optimised(ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep))
# plt.legend()
# plt.xlabel('Time / $s$')
# plt.ylabel('Muon spin polarisation')
plt.show()