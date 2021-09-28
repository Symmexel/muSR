import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, h, mu_0
from matplotlib import pyplot as plt
from numpy.fft import fft, rfft

def adj(a):
    return a.conj().T

def tp(*args):
    result = args[0]
    for t in args[1:]:
        result = np.kron(result, t)
    return result

FLUORINE_19_GYROMAGNETIC_RATIO = 251.662e6
MUON_GYROMAGNETIC_RATIO_OVER_2_PI = 135.5e6

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

class Model:
    def __init__(self, H):
        particle_count = int(np.log2(H.shape[0]))
        self.energies, self.M = eigh(H)
        self.transition_amplitude_matrix = (
            np.abs(adj(self.M) @ tp(sigma[0], *[I]*(particle_count-1)) @ self.M)**2 +
            np.abs(adj(self.M) @ tp(sigma[1], *[I]*(particle_count-1)) @ self.M)**2 +
            np.abs(adj(self.M) @ tp(sigma[2], *[I]*(particle_count-1)) @ self.M)**2
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
    def plot_cosine_amplitudes(self, ax, colour='blue'):
        N = len(self.energies)
        frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / h

        cos_amplitudes = {}

        for i in range(N):
            for j in range(N):
                cos_amplitudes[frequency_matrix[i, j].round()] = \
                    cos_amplitudes.get(frequency_matrix[i, j].round(), 0) + self.transition_amplitude_matrix[i, j]

        for (f, a) in cos_amplitudes.items():
            # ax.axvline(f, 0, a, color=colour)
            ax.plot([f/1e6, f/1e6], [0, a], color=colour, linestyle='-')

        ax.set_title('Frequency Domain')
        ax.set_xlabel('Frequency / $MHz$')
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

DISTANCE = 2.34e-10/2

H = Hamiltonian()
H.add_nn_interactions(H.add_fluorine(np.array([0, 0, DISTANCE])), DISTANCE*1.5)
H.add_nn_interactions(H.add_fluorine(np.array([0, 0, -DISTANCE])), DISTANCE*1.5)
# H.add_nn_interactions(H.add_fluorine(np.array([0, 2*DISTANCE, 0])), DISTANCE*3)
# H.add_nn_interactions(H.add_fluorine(np.array([0, -2*DISTANCE, 0])), DISTANCE*3)
# H.add_nn_interactions(H.add_fluorine(np.array([0, 2*DISTANCE, DISTANCE])), DISTANCE*3)
# H.add_nn_interactions(H.add_fluorine(np.array([0, -2*DISTANCE, -DISTANCE])), DISTANCE*3)
# H.add_nn_interactions(H.add_fluorine(np.array([0, 0, 2*DISTANCE])), DISTANCE*1.5)
# H.add_nn_interactions(H.add_fluorine(np.array([0, 2*DISTANCE, 0])), DISTANCE*2.2)
# H.add_nn_interactions(H.add_fluorine(np.array([0, 0, 2*DISTANCE])), DISTANCE*1.5)
# H.add_nn_interactions(H.add_fluorine(np.array([0, 0, -2*DISTANCE])), DISTANCE*1.5)
# H.add_nn_interactions(H.add_fluorine(np.array([DISTANCE, 0, 0])), DISTANCE*1.5)
# H.add_nn_interactions(H.add_fluorine(np.array([2*DISTANCE, 0, 0])), DISTANCE*2.2)
# H.add_nn_interactions(H.add_fluorine(np.array([-DISTANCE, 0, 0])), DISTANCE*1.5)
m = Model(H.build())
m.plot_energy_levels(plt.gca())
m.print_details()
# m.plot_energy_levels(plt.gca())
# plt.figure()

# v = tp(sigma[0], I, I) @ m.M[:, 0]
# print(m.M[:, 0])
# print(v)
# print(((adj(v) @ H.build() @ v).real/h).round())

# ts = np.arange(0, 10e-6, 1e-8)
# f = 1.5e6
# N = 50e-6*f
# ts = np.linspace(0, N/f, N)
# plt.title('Muon spin polarisation')
# plt.plot(ts, m.polarisation(ts))
# # plt.plot(ts, np.sin(2*np.pi*500_000*ts))
# plt.xlabel('Time / $s^{-1}$')
# plt.ylabel('Polarisation')
# plt.figure()

# m.plot_cosine_amplitudes(plt.gca())
# pf = np.fft.rfft(m.polarisation(ts))
# plt.plot(
#     np.linspace(0, f/2, N//2 + 1),
#     np.abs(np.concatenate(([pf[0]], 2*pf[1:])))/N
# )

plt.show()