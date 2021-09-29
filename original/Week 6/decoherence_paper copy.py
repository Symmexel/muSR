import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, h, mu_0
from matplotlib import pyplot as plt
from itertools import product

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
    def add_nn_interactions(self, i, r):
        if i == 0:
            a = np.array([0, 0, 0])
        else:
            a = self.fluorines[i]
            if norm(a) <= r:
                self.add_interaction(0, i)
        for j in range(1, self.particle_count):
            b = self.fluorines[j]
            if norm(b - a) <= r:
                self.add_interaction(i, j)
    def add_all_interactions(self, i):
        self.add_interaction(0, i)
        for j in range(1, self.particle_count):
            self.add_interaction(i, j)
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

class Model:
    def __init__(self, H):
        particle_count = int(np.log2(H.shape[0]))
        self.energies, self.M = eigh(H)
        self.transition_amplitude_matrix = (
            2*np.abs(adj(self.M) @ tp(sigma[0], *[I]*(particle_count-1)) @ self.M)**2
            + np.abs(adj(self.M) @ tp(sigma[2], *[I]*(particle_count-1)) @ self.M)**2
        ) / 3 / 2**particle_count
        self.rounded_frequencies = np.round(self.energies / h)
    def print_details(self):
        up_arrow = '\u2191'
        down_arrow = '\u2193'
        N = len(self.energies)
        particle_count = int(np.log2(len(self.energies)))

        J = sum(
            (
                np.array([
                    tp(*[I]*i, s, *[I]*(particle_count-i-1)) for s in S
                ]) for i in range(particle_count)
            ),
            start=np.zeros((N, N))
        ) / hbar
        J2 = sum(J[i] @ J[i] for i in range(3))
        Jz = J[2]

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
            # print(f'E={self.rounded_frequencies[i]} Hz')
            print(
                f'j(j+1)={(adj(self.M[:, i]) @ J2 @ self.M[:, i]).real.round(2)}'
                f', m={(adj(self.M[:, i]) @ Jz @ self.M[:, i]).real.round(2)}'
                f', E={self.rounded_frequencies[i]} Hz'
            )
    def polarisation(self, ts):
        N = len(self.energies)
        angular_frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / hbar
        return sum(
            self.transition_amplitude_matrix[i, j] * np.cos(angular_frequency_matrix[i, j]*ts)
            for i in range(N) for j in range(N)
        )
    def find_frequencies(self, cos_amplitudes, factor):
        N = len(self.energies)
        frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / h

        for i in range(N):
            for j in range(N):
                cos_amplitudes[frequency_matrix[i, j].round()] = \
                    cos_amplitudes.get(frequency_matrix[i, j].round(), 0) + factor*self.transition_amplitude_matrix[i, j]
    def plot_cosine_amplitudes(self, ax, colour=None):
        N = len(self.energies)
        frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / h

        cos_amplitudes = {}

        for i in range(N):
            for j in range(N):
                cos_amplitudes[frequency_matrix[i, j].round()] = \
                    cos_amplitudes.get(frequency_matrix[i, j].round(), 0) + self.transition_amplitude_matrix[i, j]

        for (f, a) in cos_amplitudes.items():
            ax.axvline(f, 0, a, color=colour)

        ax.set_title('Frequency Domain')
        ax.set_xlabel('Frequency / $Hz$')
        ax.set_ylabel('Amplitude')
    def plot_energy_levels(self, ax, colour=None):
        N = len(self.energies)
        rounded_frequencies = np.round(self.energies / h)
        frequency_levels = set(rounded_frequencies)
        transitions = set()
        for i in range(N):
            for j in range(N):
                if self.transition_amplitude_matrix[i, j] > 1e-8:
                    a = rounded_frequencies[i]
                    b = rounded_frequencies[j]
                    if a > b:
                        a, b = b, a
                    if a != b:
                        transitions.add((a, b))
        for f in frequency_levels:
            ax.axhline(f, 0, len(transitions)+1, color=colour)
        x = 1
        for (a, b) in transitions:
            ax.annotate('', (x, a), (x, b), arrowprops={'arrowstyle': '<->'})
            x += 1
        ax.set_title('Muon spin energy levels')
        ax.set_xlim(0, len(transitions)+1)
        ax.set_ylabel('Frequency / $Hz$')
        ax.get_xaxis().set_visible(False)

MUON_NN_FLUORINE_DISTANCE = 1.172e-10
a = 5.451e-10

nnn_muon_attraction_factor = (norm([a/2, 0, a/4]) - 0.03e-10) / norm([a/2, 0, a/4])

H = Hamiltonian()
H.add_all_interactions(H.add_fluorine(np.array([0, 0, MUON_NN_FLUORINE_DISTANCE])))
H.add_all_interactions(H.add_fluorine(np.array([0, 0, -MUON_NN_FLUORINE_DISTANCE])))
m = Model(H.build())

H.add_all_interactions(H.add_fluorine(np.array([a/2, 0, a/4])))
H.add_all_interactions(H.add_fluorine(np.array([a/2, 0, -a/4])))
H.add_all_interactions(H.add_fluorine(np.array([-a/2, 0, a/4])))
H.add_all_interactions(H.add_fluorine(np.array([-a/2, 0, -a/4])))
H.add_all_interactions(H.add_fluorine(np.array([a/2, a/4, 0])))
H.add_all_interactions(H.add_fluorine(np.array([a/2, -a/4, 0])))
H.add_all_interactions(H.add_fluorine(np.array([-a/2, a/4, 0])))
H.add_all_interactions(H.add_fluorine(np.array([-a/2, -a/4, 0])))

central_unit_cell_positions = np.array([
    [0, 0, MUON_NN_FLUORINE_DISTANCE],
    [0, 0, -MUON_NN_FLUORINE_DISTANCE],
    [a/2, 0, a/4],
    [a/2, 0, -a/4],
    [0, a/2, a/4],
    [0, a/2, -a/4],
    [a/2, a/2, a/4],
    [a/2, a/2, -a/4],
])

up_to_nnn_positions = np.array([
    # (0, 0, 0) unit cell
    [0, 0, MUON_NN_FLUORINE_DISTANCE],
    [0, 0, -MUON_NN_FLUORINE_DISTANCE],
    nnn_muon_attraction_factor*np.array([a/2, 0, a/4]),
    nnn_muon_attraction_factor*np.array([a/2, 0, -a/4]),
    nnn_muon_attraction_factor*np.array([0, a/2, a/4]),
    nnn_muon_attraction_factor*np.array([0, a/2, -a/4]),
    [a/2, a/2, a/4],
    [a/2, a/2, -a/4],

    # (-1, 0, 0) unit cell
    [-a, 0, a/4],
    [-a, 0, -a/4],
    nnn_muon_attraction_factor*np.array([-a/2, 0, a/4]),
    nnn_muon_attraction_factor*np.array([-a/2, 0, -a/4]),
    [-a, a/2, a/4],
    [-a, a/2, -a/4],
    [-a/2, a/2, a/4],
    [-a/2, a/2, -a/4],

    # (0, -1, 0) unit cell
    [0, -a, a/4],
    [0, -a, -a/4],
    [a/2, -a, a/4],
    [a/2, -a, -a/4],
    nnn_muon_attraction_factor*np.array([0, -a/2, a/4]),
    nnn_muon_attraction_factor*np.array([0, -a/2, -a/4]),
    [a/2, -a/2, a/4],
    [a/2, -a/2, -a/4],
])

radial_unit_cell_positions = np.array([
    [0, 0, a/4],
    [0, 0, -a/4],
    [a/2, 0, a/4],
    [a/2, 0, -a/4],
    [0, a/2, a/4],
    [0, a/2, -a/4],
    [a/2, a/2, a/4],
    [a/2, a/2, -a/4],
])

def second_moment_infinity(n):
    factor = (2/3) * (mu_0/(4*np.pi))**2 * hbar**2 * MUON_GYROMAGNETIC_RATIO**2 * FLUORINE_19_GYROMAGNETIC_RATIO**2 * (3/4)
    # result = 0
    result = sum(pow(norm(r), -6) for r in up_to_nnn_positions)
    for (i, j, k) in product(range(-n, n+1), repeat=3):
        # if i == j == k == 0:
        #     result += sum(pow(norm(r), -6) for r in central_unit_cell_positions)
        if (i, j, k) == (0, 0, 0) or (i, j, k) == (-1, 0, 0) or (i, j, k) == (0, -1, 0):
            pass
        else:
            result += sum(pow(norm(r), -6) for r in radial_unit_cell_positions + a*np.array([i, j, k]))
    return factor * result

def second_moment_infinity_(N):
    factor = (2/3) * (mu_0/(4*np.pi))**2 * hbar**2 * MUON_GYROMAGNETIC_RATIO**2 * FLUORINE_19_GYROMAGNETIC_RATIO**2 * (3/4)
    result = sum(pow(norm(r), -6) for r in central_unit_cell_positions)
    for n in range(1, N+1):
        for (i, j, k) in product(range(-n, n+1), repeat=3):
            if abs(i) == n or abs(j) == n or abs(k) == n:
                result += sum(pow(norm(r), -6) for r in radial_unit_cell_positions + a*np.array([i, j, k]))
    return factor * result

from scipy.optimize import curve_fit
scaling = curve_fit(H.second_moment, MUON_NN_FLUORINE_DISTANCE, second_moment_infinity(10), 0.93)[0][0]
print(scaling)

H_scaled = Hamiltonian()
H_scaled.add_all_interactions(H_scaled.add_fluorine(np.array([0, 0, MUON_NN_FLUORINE_DISTANCE])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(np.array([0, 0, -MUON_NN_FLUORINE_DISTANCE])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([a/2, 0, a/4])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([a/2, 0, -a/4])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([-a/2, 0, a/4])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([-a/2, 0, -a/4])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([a/2, a/4, 0])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([a/2, -a/4, 0])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([-a/2, a/4, 0])))
H_scaled.add_all_interactions(H_scaled.add_fluorine(scaling*np.array([-a/2, -a/4, 0])))
H_scaled.build()

# from scipy.optimize import minimize_scalar
# smi = second_moment_infinity(10)
# print(minimize_scalar(lambda scaling: abs(H.second_moment(MUON_NN_FLUORINE_DISTANCE, scaling) - smi), bounds=(0.9, 1), method='bounded'))

# m2 = Model(H.build())

# ts = np.arange(0, 20e-6, 1e-8)
# plt.title('Muon spin polarisation')
# plt.plot(ts, m.polarisation(ts), linestyle='--')
# plt.plot(ts, Model(H.build()).polarisation(ts))
# plt.plot(ts, Model(H_scaled.build()).polarisation(ts))
# plt.xlabel('Time / $s^{-1}$')
# plt.ylabel('Polarisation')

# # plt.plot(np.arange(1, 20+1), second_moment_infinity(20))

# plt.show()