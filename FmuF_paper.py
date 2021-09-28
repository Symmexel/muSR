import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, h, mu_0
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
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
    return H_scaled.build()


def zeeman_term(particle_count, B, B_axis=np.array([0, 0, 1])):
    Sn = np.tensordot(S, B_axis, axes=(0, 0))
    return -B*(
        MUON_GYROMAGNETIC_RATIO*tp(Sn, *[I]*(particle_count-1))
        + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), Sn, *[I]*(particle_count-j-1)) for j in range(1, particle_count))
    )

class FixedModelWithB:
    def __init__(self, H, particle_count, B=0, q=np.array([0, 0, 1]), a=None):
        if a is None:
            a = q
        Sn = np.tensordot(S, q, axes=(0, 0))
        self.energies, self.M = eigh(H + zeeman_term(particle_count, B, q))
        self.transition_amplitude_matrix = (
            np.abs(adj(self.M) @ tp(2*np.tensordot(sigma/2, a, axes=(0, 0)), *[I]*(particle_count-1)) @ self.M)**2
        ) / 2**particle_count
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
        frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / hbar
        return sum(
            self.transition_amplitude_matrix[i, j] * np.cos(frequency_matrix[i, j]*ts)
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
        ax.set_ylabel('Frequency / $GHz$')
        ax.get_xaxis().set_visible(False)

class AverageModel:
    def __init__(self, models):
        self.models = models
    def polarisation(self, ts):
        return sum(m.polarisation(ts) for m in self.models) / len(self.models)
    def plot_cosine_amplitudes(self, ax, colour=None):
        cos_amplitudes = {}
        for m in self.models:
            m.find_frequencies(cos_amplitudes, 1/len(self.models))
        
        # for f, a in sorted(list(cos_amplitudes.items())):
        #     if a > 1e-3:
        #         print(f / 1e6, a)

        for (f, a) in cos_amplitudes.items():
            if a > 1e-3:
                # ax.axvline(f / 1e6, 0, a, color=colour)
                ax.plot([f / 1e6, f / 1e6], [0, a**2], color='blue', linestyle='-')

        # ax.set_title('Frequency Domain')
        ax.set_xlabel('Frequency / $MHz$')
        ax.set_ylabel('Power')
        ax.set_yscale('log')
        # ax.set_ylabel('Amplitude')

B = 220e-4
# B = 220e-3
# B = 220e-2
# r = 1.16e-10
r = 1.17e-10

MU_F_DISTANCE = 1.17e-10
a = 4.62e-10

nn1 = np.array([
    [MU_F_DISTANCE/np.sqrt(2), MU_F_DISTANCE/np.sqrt(2), 0],
    [-MU_F_DISTANCE/np.sqrt(2), -MU_F_DISTANCE/np.sqrt(2), 0]
])
nnn1 = np.array([
    [a/8, -a/8, a/4],
    [a/8, -a/8, -a/4],
    [-a/8, a/8, a/4],
    [-a/8, a/8, -a/4]
])
radial1 = np.array([
    [a/8, a/8, 0],
    [-a/8, -a/8, 0],
    [a/8, -a/8, a/4],
    [a/8, -a/8, -a/4],
    [-a/8, a/8, a/4],
    [-a/8, a/8, -a/4]
])

nn1 = np.array([
    [MU_F_DISTANCE/np.sqrt(2), MU_F_DISTANCE/np.sqrt(2), 0],
    [-MU_F_DISTANCE/np.sqrt(2), -MU_F_DISTANCE/np.sqrt(2), 0]
])
nnn1 = np.array([
    [a/8, -a/8, a/4],
    [a/8, -a/8, -a/4],
    [-a/8, a/8, a/4],
    [-a/8, a/8, -a/4]
])
radial1 = np.array([
    [a/8, a/8, 0],
    [-a/8, -a/8, 0],
    [a/8, -a/8, a/4],
    [a/8, -a/8, -a/4],
    [-a/8, a/8, a/4],
    [-a/8, a/8, -a/4]
])

nn2 = np.array([
    [r[0], -r[1], r[2]] for r in nn1
])
nnn2 = np.array([
    [r[0], -r[1], r[2]] for r in nnn1
])
radial2 = np.array([
    [r[0], -r[1], r[2]] for r in radial1
])

nn3 = np.array([
    [r[0], r[2], r[1]] for r in nn1
])
nnn3 = np.array([
    [r[0], r[2], r[1]] for r in nnn1
])
radial3 = np.array([
    [r[0], r[2], r[1]] for r in radial1
])

nn4 = np.array([
    [r[0], r[2], -r[1]] for r in nn1
])
nnn4 = np.array([
    [r[0], r[2], -r[1]] for r in nnn1
])
radial4 = np.array([
    [r[0], r[2], -r[1]] for r in radial1
])

nn5 = np.array([
    [r[2], r[1], r[0]] for r in nn1
])
nnn5 = np.array([
    [r[2], r[1], r[0]] for r in nnn1
])
radial5 = np.array([
    [r[2], r[1], r[0]] for r in radial1
])

nn6 = np.array([
    [r[2], -r[1], r[0]] for r in nn1
])
nnn6 = np.array([
    [r[2], -r[1], r[0]] for r in nnn1
])
radial6 = np.array([
    [r[2], -r[1], r[0]] for r in radial1
])

# B || <100>
m_100 = AverageModel([
    FixedModelWithB(scale(nn1, nnn1, radial1, np.array([0, 0, 1])), 7, B, np.array([0, 0, 1]), np.array([1, 0, 0])),
    FixedModelWithB(scale(nn2, nnn2, radial2, np.array([0, 0, 1])), 7, B, np.array([0, 0, 1]), np.array([1, 0, 0])),
    FixedModelWithB(scale(nn3, nnn3, radial3, np.array([0, 0, 1])), 7, B, np.array([0, 0, 1]), np.array([1, 0, 0])),
    FixedModelWithB(scale(nn4, nnn4, radial4, np.array([0, 0, 1])), 7, B, np.array([0, 0, 1]), np.array([1, 0, 0])),
    FixedModelWithB(scale(nn5, nnn5, radial5, np.array([0, 0, 1])), 7, B, np.array([0, 0, 1]), np.array([1, 0, 0])),
    FixedModelWithB(scale(nn6, nnn6, radial6, np.array([0, 0, 1])), 7, B, np.array([0, 0, 1]), np.array([1, 0, 0])),
])

# B || <110>
q = np.array([1, 1, 0])/np.sqrt(2)
a = np.array([-1, 1, 0])/np.sqrt(2)
m_110 = AverageModel([
    FixedModelWithB(scale(nn1, nnn1, radial1, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn2, nnn2, radial2, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn3, nnn3, radial3, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn4, nnn4, radial4, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn5, nnn5, radial5, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn6, nnn6, radial6, q), 7, B, q, np.array([1, 0, 0])),
])

# B || <111>
q = np.array([1, 1, 1])/np.sqrt(3)
a = np.array([2, -1, -1])/np.sqrt(6)
# a = np.array([0, 1, -1])/np.sqrt(2)
m_111 = AverageModel([
    FixedModelWithB(scale(nn1, nnn1, radial1, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn2, nnn2, radial2, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn3, nnn3, radial3, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn4, nnn4, radial4, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn5, nnn5, radial5, q), 7, B, q, np.array([1, 0, 0])),
    FixedModelWithB(scale(nn6, nnn6, radial6, q), 7, B, q, np.array([1, 0, 0])),
])

# print(mu_0 * FLUORINE_19_GYROMAGNETIC_RATIO * MUON_GYROMAGNETIC_RATIO_OVER_2_PI / (2*r**3) * (hbar)**2 / h / 1e6)

# m = FixedModelWithB(
#     MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*B*S[2],
#     1, 0, np.array([0, 0, 1]), np.array([1, 0, 0])
# )

# ts = np.arange(0, 10e-6, 1e-8)
ts = np.arange(0, 7e-6, 1e-8)
# plt.title('Muon spin polarisation')
# plt.plot(ts, m_111.polarisation(ts))
# plt.xlabel('Time / $s^{-1}$')
# plt.ylabel('Polarisation, $D(t)$')

# plt.figure()
# # plt.yscale('log')
# m.plot_cosine_amplitudes(plt.gca())

m_111_polarisation = m_111.polarisation(ts)

fig, ax = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={
    'wspace': 0.4,
    'hspace': 0.3
})
ax[0][0].plot(ts*1e6, m_111_polarisation, color='blue')
ax[0][0].set_xlabel('Time / ${\mu}s$')
ax[0][0].set_ylabel('Polarisation, $D(t)$')
# ax[0][0].set_title('(a)')
ax[0][0].annotate('(a)', (0.95, 0.9), xycoords='axes fraction', ha='right')
m_111.plot_cosine_amplitudes(ax[0][1])
# ax[0][1].set_title('(b)')
ax[0][1].annotate('(b)', (0.05, 0.9), xycoords='axes fraction')
m_110.plot_cosine_amplitudes(ax[1][0])
# ax[1][0].set_title('(c)')
ax[1][0].annotate('(c)', (0.05, 0.9), xycoords='axes fraction')
m_100.plot_cosine_amplitudes(ax[1][1])
# ax[1][1].set_title('(d)')
ax[1][1].annotate('(d)', (0.05, 0.9), xycoords='axes fraction')
plt.show()
