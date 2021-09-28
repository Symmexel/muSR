import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, h, mu_0
from matplotlib import pyplot as plt
from numpy.random import default_rng
from itertools import product
from numpy.fft import rfft

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

def zeeman_term(particle_count, B, B_axis=np.array([0, 0, 1])):
    Sn = np.tensordot(S, B_axis, axes=(0, 0))
    return -B*(
        MUON_GYROMAGNETIC_RATIO*tp(Sn, *[I]*(particle_count-1))
        + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), Sn, *[I]*(particle_count-j-1)) for j in range(1, particle_count))
    )

class SolveH:
    def __init__(self, H, measurement_axis=np.array([0, 0, 1]), x_axis=np.array([1, 0, 0]), y_axis=np.array([0, 1, 0])):
        self.H = H
        self.measurement_axis = measurement_axis
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.particle_count = int(np.log2(self.H.shape[0]))
    def polarisation(self, ts, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        for i in range(1, len(ts)):
            energies, M = eigh(H+H1(ts[i-1], self))
            U = M @ np.diag(np.exp(energies*(-1j*(ts[i]-ts[i-1])/hbar))) @ adj(M) @ U
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D
    def polarisation_optimised_(self, ts, period, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H

        def H1_(t, model=None):
            return pulse_B * (
                zeeman_term(2, np.cos(MUON_GYROMAGNETIC_RATIO*B*t), model.x_axis)
                + zeeman_term(2, -np.sin(MUON_GYROMAGNETIC_RATIO*B*t), model.y_axis)
            )
        H_ = (
            mu_0 *
            FLUORINE_19_GYROMAGNETIC_RATIO *
            FLUORINE_19_GYROMAGNETIC_RATIO / (4*np.pi*MU_F_DISTANCE**3)
        ) * sum(tp(S[k], S[k]) for k in range(3)) - 3*tp(S[0], S[0])
        U_ = np.eye(4)
        ts_ = np.arange(-6e-8, 0, 1e-9)
        for i in range(1, len(ts_)):
            energies, M = eigh(H_+H1_(ts[i-1], self))
            U_ = M @ np.diag(np.exp(energies*(-1j*(ts_[i]-ts_[i-1])/hbar))) @ adj(M) @ U_
        print(U_)

        U = np.kron(I, U_)#np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        dt = ts[1] - ts[0]
        dU = np.ndarray((int(np.ceil(period / dt)), *U.shape), np.complex)
        for i in range(dU.shape[0]):
            energies, M = eigh(H+H1(ts[i], self))
            dU[i] = M @ np.diag(np.exp(energies*(-1j*dt/hbar))) @ adj(M)
        for i in range(1, len(ts)):
            U = dU[int((ts[i]%period)//dt)] @ U
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D
    def polarisation_optimised(self, ts, period, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        D = np.zeros(ts.shape)
        D[0] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        dt = ts[1] - ts[0]
        dU = np.ndarray((int(np.ceil(period / dt)), *U.shape), np.complex)
        for i in range(dU.shape[0]):
            energies, M = eigh(H+H1(ts[i], self))
            dU[i] = M @ np.diag(np.exp(energies*(-1j*dt/hbar))) @ adj(M)
        for i in range(1, len(ts)):
            U = dU[int((ts[i]%period)//dt)] @ U
            D[i] = np.trace(U @ sigma_mu @ adj(U) @ sigma_mu).real
        D /= N
        return D
    def transition_amplitude_matrix(self, ts, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        M = None
        for i in range(1, len(ts)):
            energies, M = eigh(H+H1(ts[i-1], self))
            U = M @ np.diag(np.exp(energies*(-1j*(ts[i]-ts[i-1])/hbar))) @ adj(M) @ U
        assert M
        return (
            np.abs(adj(M) @ U @ sigma_mu @ adj(U) @ M)
            * np.abs(adj(M) @ sigma_mu @ M)
        ) / N
    def transition_amplitude_matrix_optimised(self, ts, period, H1=lambda t, model: 0, H1_prep=None):
        if H1_prep:
            H1_prep(self)
        N = self.H.shape[0]
        sigma_mu = tp(np.tensordot(sigma, self.measurement_axis, axes=(0, 0)), *[I]*(self.particle_count-1))
        H = self.H
        U = np.eye(N)
        dt = ts[1] - ts[0]
        dU = np.ndarray((int(np.ceil(period / dt)), *U.shape), np.complex)
        for i in range(dU.shape[0]):
            energies, M = eigh(H+H1(ts[i], self))
            dU[i] = M @ np.diag(np.exp(energies*(-1j*dt/hbar))) @ adj(M)
        for i in range(1, len(ts)):
            U = dU[int((ts[i]%period)//dt)] @ U
        energies, M = eigh(H+H1(ts[-2], self))
        return (
            np.abs(adj(M) @ U @ sigma_mu @ adj(U) @ M)
            * np.abs(adj(M) @ sigma_mu @ M)
        ) / N

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

class RK4:
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
        self.thetas = thetas
        self.samples = []
        for i in range(sample_count):
            B_axis = np.array([
                np.sin(thetas[i])*np.cos(phis[i]),
                np.sin(thetas[i])*np.sin(phis[i]),
                np.cos(thetas[i])
            ])
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
            self.samples.append(fixed_model(H + zeeman_term(particle_count, B, B_axis), B_axis, x_axis, y_axis))
    def polarisation(self, *args, **kwargs):
        return sum(m.polarisation(*args, **kwargs) for m in self.samples) / len(self.samples)
    def polarisation_optimised(self, *args, **kwargs):
        return sum(m.polarisation_optimised(*args, **kwargs) for m in self.samples) / len(self.samples)
    def polarisation_optimised_(self, *args, **kwargs):
        return sum(m.polarisation_optimised_(*args, **kwargs) for m in self.samples) / len(self.samples)

B = 1583e-4
pulse_B = 18.4e-4
pulse_duration = 1e-6
pulse_angular_frequency = MUON_GYROMAGNETIC_RATIO*B
pulse2_B = FLUORINE_19_GYROMAGNETIC_RATIO*B

a = 4.62e-10
MU_F_DISTANCE = 1.17e-10

HB = Hamiltonian()
H_small = HB.build()
# HB.add_all_interactions(HB.add_fluorine(np.array([MU_F_DISTANCE/np.sqrt(2), MU_F_DISTANCE/np.sqrt(2), 0])))
# HB.add_all_interactions(HB.add_fluorine(np.array([-MU_F_DISTANCE/np.sqrt(2), -MU_F_DISTANCE/np.sqrt(2), 0])))
HB.add_all_interactions(HB.add_fluorine(np.array([MU_F_DISTANCE, 0, 0])))
HB.add_all_interactions(HB.add_fluorine(np.array([-MU_F_DISTANCE, 0, 0])))
H = HB.build()

# def H1(t, model=None):
#     if t < (np.pi/2) / (MUON_GYROMAGNETIC_RATIO*pulse_B):
#         return pulse_B * (
#             np.cos(MUON_GYROMAGNETIC_RATIO*B*t) * (
#                 MUON_GYROMAGNETIC_RATIO*tp(S[0], *[I]*(HB.particle_count-1))
#                 + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), S[0], *[I]*(HB.particle_count-j-1)) for j in range(1, HB.particle_count))
#             ) - np.sin(MUON_GYROMAGNETIC_RATIO*B*t) * (
#                 MUON_GYROMAGNETIC_RATIO*tp(S[1], *[I]*(HB.particle_count-1))
#                 + sum(FLUORINE_19_GYROMAGNETIC_RATIO*tp(*[I]*(j), S[1], *[I]*(HB.particle_count-j-1)) for j in range(1, HB.particle_count))
#             )
#         )
#     else:
#         return 0

def H1(t, model=None):
    # return pulse_B * (
    #     zeeman_term(model.particle_count, np.cos(MUON_GYROMAGNETIC_RATIO*B*t), model.x_axis)
    #     + zeeman_term(model.particle_count, -np.sin(MUON_GYROMAGNETIC_RATIO*B*t), model.y_axis)
    # )
    return (
        np.cos(MUON_GYROMAGNETIC_RATIO*B*t) * model.matrix_x
        -np.sin(MUON_GYROMAGNETIC_RATIO*B*t) * model.matrix_y
    )
    # if t < (np.pi/2) / (MUON_GYROMAGNETIC_RATIO*pulse_B):
    #     return pulse_B * (
    #         zeeman_term(particle_count, np.cos(MUON_GYROMAGNETIC_RATIO*B*t), model.x_axis)
    #         + zeeman_term(particle_count, -np.sin(MUON_GYROMAGNETIC_RATIO*B*t), model.y_axis)
    #     )
    # else:
    #     return 0

def H_pulse_prep(model):
    model.matrix_x = zeeman_term(model.particle_count, pulse_B, model.x_axis)
    model.matrix_y = zeeman_term(model.particle_count, pulse_B, model.y_axis)

def H_pulse(t, model=None):
    # return pulse_B * zeeman_term(model.particle_count, np.cos(MUON_GYROMAGNETIC_RATIO*B*t), model.x_axis)
    return np.cos(MUON_GYROMAGNETIC_RATIO*B*t) * model.matrix_x
    # if t < (np.pi/2) / (MUON_GYROMAGNETIC_RATIO*pulse_B):
    #     return pulse_B * zeeman_term(particle_count, np.cos(MUON_GYROMAGNETIC_RATIO*B*t), model.x_axis)
    # else:
    #     return 0

# ts = np.arange(0, 30e-6, 1e-9)
ts = np.arange(0, 10e-6, 1e-9)

# p = TimeDepModelWithB(H, B, 20).polarisation(ts, H1)

# pf = rfft(p)
# f = 1/ts[1]
# N = len(ts)

# plt.title(r'Powder average, isolated $\mu$')
# plt.plot(ts, p)
# plt.plot(ts, SolveH(H+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H1, H_pulse_prep), label='Rotating field in xy-plane')
# plt.plot(ts, SolveH(H+zeeman_term(HB.particle_count, B, np.array([0, 0, 1]))).polarisation(ts, H_pulse, H_pulse_prep), label='Cosine field along x-axis')
# plt.plot(ts, TimeDepModelWithB(H_small, B, 20).polarisation(ts, H_pulse), label='mu')

# plt.plot(ts, TimeDepModelWithB(H_small, B, 20).polarisation(ts, H1), label='Rotating field in xy-plane')
# plt.plot(ts, TimeDepModelWithB(H_small, B, 20).polarisation(ts, H_pulse), label='Cosine field along x-axis')
# plt.plot(ts, np.cos(MUON_GYROMAGNETIC_RATIO*pulse_B*ts), linestyle='--', label=r'$cos(\gamma_{\mu}B_{Pulse}t)$')
# plt.legend()
# plt.xlabel('Time / $s^{-1}$')
# plt.ylabel('Muon spin polarisation')

# plt.figure()

# plt.title(r'Powder average, $F{\mu}F$')
# plt.plot(ts, TimeDepModelWithB(H, B, 20).polarisation(ts, H1, H_pulse_prep), label='Rotating field in xy-plane')
# plt.plot(ts, TimeDepModelWithB(H, B, 20).polarisation(ts, H_pulse, H_pulse_prep), label='Cosine field along x-axis')
# plt.plot(ts, SolveH(H+zeeman_term(HB.particle_count, B), np.array([0, 0, 1])).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H1, H_pulse_prep)*1e6, label='Rotating field in xy-plane')
# plt.plot(ts, SolveH(H+zeeman_term(HB.particle_count, B), np.array([0, 0, 1])).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep)*1e6, label='Cosine field along x-axis')
# plt.plot(ts*1e6, TimeDepModelWithB(H_small, B, 20).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H1, H_pulse_prep), color='blue', linestyle='--', label='Rotating field in xy-plane')
# plt.plot(ts*1e6, TimeDepModelWithB(H_small, B, 20).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep), color='red', linestyle='--', label='Cosine field along x-axis')
plt.plot(ts*1e6, TimeDepModelWithB(H, B, 20).polarisation_optimised(
    ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H1, H_pulse_prep), label='Rotating field in xy-plane')
plt.plot(ts*1e6, TimeDepModelWithB(H, B, 20).polarisation_optimised_(
    ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H1, H_pulse_prep), label='Rotating field in xy-plane')
# plt.plot(ts*1e6, TimeDepModelWithB(H, B, 20).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep), color='red', label='Cosine field along x-axis')

# plt.plot(ts, np.cos(MUON_GYROMAGNETIC_RATIO*pulse_B*ts)*1e6, linestyle='--', label=r'$cos(\gamma_{\mu}B_{Pulse}t)$')
# plt.plot(ts, np.cos(0.5*MUON_GYROMAGNETIC_RATIO*pulse_B*ts)*1e6, linestyle='--', label=r'$cos(\frac{1}{2}\gamma_{\mu}B_{Pulse}t)$')
# plt.legend()
plt.xlabel('Time / ${\mu}s$')
plt.ylabel('Polarisation, $D(t)$')
plt.show()

# fig, ax = plt.subplots(1, 2, sharey=True)
# ax[0].plot(ts*1e6, TimeDepModelWithB(H_small, B, 20).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H1, H_pulse_prep), label='Rotating field')
# ax[0].plot(ts*1e6, TimeDepModelWithB(H_small, B, 20).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep), label='Oscillaing field')
# ax[0].plot(ts*1e6, np.cos(MUON_GYROMAGNETIC_RATIO*pulse_B*ts), linestyle='--', label=r'$cos(\gamma_{\mu}B_{Pulse}t)$')
# ax[0].plot(ts*1e6, np.cos(0.5*MUON_GYROMAGNETIC_RATIO*pulse_B*ts), linestyle='--', label=r'$cos(\frac{1}{2}\gamma_{\mu}B_{Pulse}t)$')

# ax[1].plot(ts*1e6, TimeDepModelWithB(H, B, 20).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H1, H_pulse_prep), label='Rotating field')
# ax[1].plot(ts*1e6, TimeDepModelWithB(H, B, 20).polarisation_optimised(
#     ts, 1/(MUON_GYROMAGNETIC_RATIO_OVER_2_PI*B), H_pulse, H_pulse_prep), label='Cosine field along x-axis')
# ax[1].plot(ts*1e6, np.cos(MUON_GYROMAGNETIC_RATIO*pulse_B*ts), linestyle='--', label=r'$cos(\gamma_{\mu}B_{Pulse}t)$')
# ax[1].plot(ts*1e6, np.cos(0.5*MUON_GYROMAGNETIC_RATIO*pulse_B*ts), linestyle='--', label=r'$cos(\frac{1}{2}\gamma_{\mu}B_{Pulse}t)$')

# ax[0].set_title(r'Isolated $\mu$')
# ax[0].set_xlabel('Time / ${\mu}s$')
# ax[0].set_ylabel('Polarisation, $D(t)$')

# ax[1].set_title(r'$F-{\mu}-F$')
# ax[1].set_xlabel('Time / ${\mu}s$')

plt.show()

# plt.figure()
# plt.plot(
#     np.linspace(0, f/2, N//2 + 1),
#     np.abs(np.concatenate(([pf[0]], 2*pf[1:])))/N
# )