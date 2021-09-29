import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, h, mu_0
from matplotlib import pyplot as plt
from numpy.random import default_rng

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
        # return np.tensordot(
        #     np.cos(np.tensordot(ts, angular_frequency_matrix, axes=0)),
        #     self.transition_amplitude_matrix,
        #     axes=2
        # )
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

class FixedModelWithB:
    def __init__(self, H, B=0, B_axis=np.array([0, 0, 1]), measurement_axis=None):
        particle_count = int(np.log2(H.shape[0]))
        if measurement_axis is None:
            measurement_axis = B_axis
        # Sn = np.tensordot(S, B_axis, axes=(0, 0))
        # self.energies, self.M = eigh(H + B*(
        #     MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*tp(Sn, I)
        #     + FLUORINE_19_GYROMAGNETIC_RATIO*tp(I, Sn)
        # ))
        self.energies, self.M = eigh(H + zeeman_term(particle_count, B, B_axis))
        self.transition_amplitude_matrix = (
            np.abs(adj(self.M) @ tp(2*np.tensordot(S, measurement_axis, axes=(0, 0))/hbar, *[I]*(particle_count-1)) @ self.M)**2
        ) / 2**particle_count
        self.rounded_frequencies = np.round(self.energies / hbar)
    def print_details(self):
        S_muon = np.array([tp(s, I) for s in S])
        S_fluorine = np.array([tp(I, s) for s in S])
        J = (S_muon + S_fluorine) / hbar
        # S_muon = np.array([tp(s, I, I) for s in S])
        # S_fluorine1 = np.array([tp(I, s, I) for s in S])
        # S_fluorine2 = np.array([tp(I, I, s) for s in S])
        # J = (S_muon + S_fluorine1 + S_fluorine2) / hbar
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
            # print(self.rounded_frequencies[i])
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
        frequency_matrix = np.abs(np.tile(self.energies, (N, 1)).T - np.tile(self.energies, (N, 1))) / hbar

        cos_amplitudes = {}

        for i in range(N):
            for j in range(N):
                cos_amplitudes[frequency_matrix[i, j].round()] = \
                    cos_amplitudes.get(frequency_matrix[i, j].round(), 0) + self.transition_amplitude_matrix[i, j]

        for (f, a) in cos_amplitudes.items():
            ax.axvline(f * 1e-6, 0, a, color=colour)

        ax.set_title('Frequency Domain')
        ax.set_xlabel('Frequency / $GHz$')
        ax.set_ylabel('Amplitude')
    def plot_energy_levels(self, ax, colour=None):
        N = len(self.energies)
        rounded_frequencies = np.round(self.energies / hbar)
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
            ax.axhline(f/1e6, 0, len(transitions)+1, color=colour)
        x = 1
        for (a, b) in transitions:
            ax.annotate('', (x, a/1e6), (x, b/1e6), arrowprops={'arrowstyle': '<->'})
            x += 1
        ax.set_title('Muon spin energy levels')
        ax.set_xlim(0, len(transitions)+1)
        ax.set_ylabel('Frequency / $GHz$')
        ax.get_xaxis().set_visible(False)

def zeeman_term(particle_count, B, B_axis):
    Sn = np.tensordot(S, B_axis, axes=(0, 0))
    return -B*(
        MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*tp(Sn, *[I]*(particle_count-1))
        + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), Sn, *[I]*(particle_count-j-1)) for j in range(1, particle_count))
    )

class SolveH:
    def __init__(self, H, measurement_axis=np.array([0, 0, 1])):
        self.H = H
        self.measurement_axis = measurement_axis
    def polarisation(self, ts, H1=lambda t: 0):
        N = self.H.shape[0]
        particle_count = int(np.log2(N))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        for i in range(1, len(ts)):
            energies, M = eigh(H+H1(ts[i-1]))
            U = M @ np.diag(np.exp(energies*(-1j*(ts[i]-ts[i-1])/hbar))) @ adj(M) @ U
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
            # D[i] = np.trace(U @ sigma_mu @ adj(U) @ tp(np.tensordot(sigma, self.x_axis, axes=(0, 0)), *[I]*(particle_count-1))).real
        D /= N
        return D

class ForwardEuler:
    def __init__(self, H, measurement_axis=np.array([0, 0, 1])):
        self.H = H
        self.measurement_axis = measurement_axis
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
    def __init__(self, H, measurement_axis=np.array([0, 0, 1])):
        self.H = H
        self.measurement_axis = measurement_axis
    def polarisation(self, ts, H1=lambda t: 0):
        N = self.H.shape[0]
        particle_count = int(np.log2(N))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        for i in range(1, len(ts)):
            U = np.linalg.solve(
                np.eye(N) + (1j * (ts[i] - ts[i-1]) / 2 / hbar)*(H+H1(ts[i])),
                (np.eye(N) - (1j * (ts[i] - ts[i-1]) / 2 / hbar)*(H+H1(ts[i-1]))) @ U
            )
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D

class RK4:
    def __init__(self, H, measurement_axis=np.array([0, 0, 1])):
        self.H = H
        self.measurement_axis = measurement_axis
    def polarisation(self, ts, H1=lambda t: 0):
        N = self.H.shape[0]
        particle_count = int(np.log2(N))
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        for i in range(1, len(ts)):
            U1 = U - (1j * (ts[i] - ts[i-1]) / hbar) * (H+H1(ts[i-1])) @ U
            U2 = U - (1j * (ts[i] - ts[i-1]) / hbar) * (H+H1(ts[i])) @ U1
            U = (U1 + U2) / 2
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D

class ModelWithB:
    def __init__(self, H, B=0, sample_count=10):
        # thetas = np.arccos(1 - 2*np.array([rng.random(sample_count//10)/10 + i/10 for i in range(10)]).flatten())
        # phis = 2*np.pi*rng.random(sample_count)
        thetas = thetas_
        phis = phis_
        self.thetas = thetas
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
        # thetas = np.arccos(1 - 2*np.array([rng.random(sample_count//10)/10 + i/10 for i in range(10)]).flatten())
        # phis = 2*np.pi*rng.random(sample_count)
        thetas = thetas_
        phis = phis_
        self.thetas = thetas
        self.samples = []
        for i in range(sample_count):
            B_axis = np.array([
                np.sin(thetas[i])*np.cos(phis[i]),
                np.sin(thetas[i])*np.sin(phis[i]),
                np.cos(thetas[i])
            ])
            self.samples.append(fixed_model(H + zeeman_term(particle_count, B, B_axis), B_axis))
    def polarisation(self, *args, **kwargs):
        return sum(m.polarisation(*args, **kwargs) for m in self.samples) / len(self.samples)

thetas_ = np.arccos(1 - 2*np.array([rng.random(20//10)/10 + i/10 for i in range(10)]).flatten())
phis_ = 2*np.pi*rng.random(20)

r = 1.17e-10
# B = 220e-4
# B_pulse = 1e-3
B = 1583e-4
B_pulse = 18.4e-4

HB = Hamiltonian()
H_small = HB.build()
HB.add_all_interactions(HB.add_fluorine(np.array([0, r, 0])))
# HB.add_all_interactions(HB.add_fluorine(np.array([0, 0, -r])))
# HB.add_all_interactions(HB.add_fluorine(np.array([0, r, 0])))
# HB.add_all_interactions(HB.add_fluorine(np.array([0, -r, 0])))
# HB.add_all_interactions(HB.add_fluorine(np.array([r, 0, 0])))

H0 = HB.build()

def H1_(t):
    if t < (np.pi/2) / (MUON_GYROMAGNETIC_RATIO*B_pulse):
        return B_pulse * (
            np.cos(MUON_GYROMAGNETIC_RATIO*B*t) * (
                MUON_GYROMAGNETIC_RATIO*S[0]
            ) - 0*np.sin(MUON_GYROMAGNETIC_RATIO*B*t) * (
                MUON_GYROMAGNETIC_RATIO*S[1]
            )
        )
    else:
        return 0

def H2_(t):
    if t < (np.pi/2) / (MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*B_pulse):
        return H1_(t)
    else:
        return 1000*B_pulse * (
            np.cos(MUON_GYROMAGNETIC_RATIO*B*t) * (
                MUON_GYROMAGNETIC_RATIO*S[1]
            )
        )

def H1(t):
    if t < (np.pi/2) / (MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*B_pulse):
        return B_pulse * (
            np.cos(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*B*t) * (
                MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*tp(S[0], *[I]*(HB.particle_count-1))
                + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), S[0], *[I]*(HB.particle_count-j-1)) for j in range(1, HB.particle_count))
            ) - 0*np.sin(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*B*t) * (
                MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*tp(S[1], *[I]*(HB.particle_count-1))
                + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), S[1], *[I]*(HB.particle_count-j-1)) for j in range(1, HB.particle_count))
            )
        )
    else:
        return 0

def H2(t):
    if t < (np.pi/2) / (MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*B_pulse):
        return H1(t)
    else:
        return 1000*B_pulse * (
            np.cos(FLUORINE_19_GYROMAGNETIC_RATIO*2*np.pi*B*t) * (
                MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*tp(S[1], *[I]*(HB.particle_count-1))
                + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), S[1], *[I]*(HB.particle_count-j-1)) for j in range(1, HB.particle_count))
            )
        )

ts = np.arange(0, 10e-6, 1e-9)
# ts = np.arange(0, 1e-8, 1e-12)
# ts = np.arange(0, 26e-6, 1e-9)
# ts = np.arange(0, 10e-6, 1e-8)
ts2 = np.arange(0, 10e-6, 1e-8)

# p = SolveH(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H1)

plt.title('Muon spin polarisation')
# plt.plot(ts, SolveH(H_small+zeeman_term(1, B, np.array([0, 0, 1]))).polarisation(ts, H2_), label=r'${\mu}+{\tau}_{90}+Decoupling$')
# plt.plot(ts, SolveH(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H2), label=r'$F{\mu}F+{\tau}_{90}+Decoupling$')
# plt.plot(ts, SolveH(H_small+zeeman_term(1, B, np.array([0, 0, 1]))).polarisation(ts, H1_), label=r'${\mu}+{\tau}_{90}$')
# plt.plot(ts, SolveH(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H1), label=r'$F{\mu}F+{\tau}_{90}$')

# plt.plot(ts, SolveH(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H_decoupling), label='SolveH')
# plt.plot(ts2, SolveH(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts2, H1), label='SolveH')
# plt.plot(ts, np.cos(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*B*ts), linestyle='--')
plt.plot(ts, SolveH(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H1), label='SolveH')
plt.plot(ts, CN(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H1), label='CN')
plt.plot(ts, RK4(H0+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H1), label='Heun')
# plt.plot(ts2, SolveH(H0+zeeman_term(int(np.log2(H0.shape[0])), B, np.array([0, 0, 1]))).polarisation(ts2, H1), linestyle='--')
# plt.plot(ts2, CN(H0+zeeman_term(int(np.log2(H0.shape[0])), B, np.array([0, 0, 1]))).polarisation(ts2, H1), linestyle='--')
# plt.plot(ts2, RK4(H0+zeeman_term(int(np.log2(H0.shape[0])), B, np.array([0, 0, 1]))).polarisation(ts2, H1), linestyle='--')
# plt.plot(ts, SolveH(H0+zeeman_term(int(np.log2(H0.shape[0])), B, np.array([0, 0, 1]))).polarisation(ts, lambda t: H1(t+1e-8/2)))
# plt.plot(ts, SolveH(H0+zeeman_term(int(np.log2(H0.shape[0])), B, np.array([0, 0, 1]))).polarisation(ts, lambda t: H1(t+1e-8)))
# plt.plot(ts, TimeDepModelWithB(H0, B, 20).polarisation(ts, H1), label='MC SolveH')
# plt.plot(ts, TimeDepModelWithB(H0, B, 20, CN).polarisation(ts, H1), label='MC CN')
# plt.plot(ts, TimeDepModelWithB(H0, B, 20, RK4).polarisation(ts, H1), label='MC Heun')
# plt.plot(ts, TimeDepModelWithB(H0, B, 20).polarisation(ts), label='MC SolveH')
# plt.plot(ts, TimeDepModelWithB(H0, B, 20, CN).polarisation(ts), label='MC CN')
# plt.plot(ts, TimeDepModelWithB(H0, B, 20, RK4).polarisation(ts), label='MC Heun')
# plt.plot(ts, ModelWithB(H0, B, 20).polarisation(ts), label='MC', linestyle='--')
# plt.plot(ts, Model(H0+zeeman_term(int(np.log2(H0.shape[0])), B, np.array([0, 0, 1]))).polarisation(ts), linestyle='--', label='Exact')
# plt.plot(ts, Model(H0).polarisation(ts), linestyle='--', label='Exact')
plt.xlabel('Time / $s^{-1}$')
plt.ylabel('Polarisation')
plt.legend()

plt.show()