import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, mu_0
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

I = np.eye(2)
sigma_x = np.array([
    [0, 1],
    [1, 0]
])
sigma_z = np.array([
    [1, 0],
    [0, -1]
])
S = (hbar/2)*np.array([
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
        [0, -1],
    ]
])

rng = default_rng()

class Model:
    def __init__(self, H, particle_count):
        self.energies, self.M = eigh(H)
        self.transition_amplitude_matrix = (
            2 * (np.abs(adj(self.M) @ tp(sigma_x, *[I]*(particle_count-1)) @ self.M)**2)
            + np.abs(adj(self.M) @ tp(sigma_z, *[I]*(particle_count-1)) @ self.M)**2
        ) / 3 / 2**particle_count
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
                f', jz={(adj(self.M[:, i]) @ Jz @ self.M[:, i]).real.round(2)}'
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

        ax.set_title('Cosine amplitudes')
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

class FixedModelWithB:
    def __init__(self, H, B=0, B_axis=np.array([0, 0, 1]), measurement_axis=None):
        particle_count = int(np.log2(H.shape[0]))
        if measurement_axis is None:
            measurement_axis = B_axis
        Sn = np.tensordot(S, B_axis, axes=(0, 0))
        self.energies, self.M = eigh(H + B*(
            MUON_GYROMAGNETIC_RATIO_OVER_2_PI*2*np.pi*tp(Sn, I)
            + FLUORINE_19_GYROMAGNETIC_RATIO*tp(I, Sn)
        ))
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

class Test:
    def __init__(self, H, particle_count, B=0):
        self.samples = []
        for i in range(3):
            q = np.eye(3)[i]
            self.samples.append(FixedModelWithB(H, B, q))
    
    def polarisation(self, ts):
        return sum(m.polarisation(ts) for m in self.samples) / 3

class ModelWithB:
    def __init__(self, H, particle_count, B=0, sample_count=1):
        # thetas = np.pi*rng.random(sample_count)
        # thetas = np.arccos(1 - 2*rng.random(sample_count))
        # thetas = np.pi*np.array([rng.random(4)/5 + i/5 for i in range(5)]).flatten()
        # thetas = np.arccos(1 - 2*np.array([rng.random(5)/4 + i/4 for i in range(4)]).flatten())
        # thetas = np.arccos(1 - 2*np.array([rng.random(4)/5 + i/5 for i in range(5)]).flatten())
        # thetas = np.arccos(1 - 2*np.array([rng.random(2)/10 + i/10 for i in range(10)]).flatten())
        thetas = np.arccos(1 - 2*np.array([rng.random(sample_count//10)/10 + i/10 for i in range(10)]).flatten())
        # thetas = np.arccos(1 - 2*np.array([rng.random(25)/4 + i/4 for i in range(4)]).flatten())
        phis = 2*np.pi*rng.random(sample_count)
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
        # return sum(m.polarisation(ts)*np.sin(self.thetas[i]) for i, m in enumerate(self.samples)) * np.pi/2 / len(self.samples)
        return sum(m.polarisation(ts) for m in self.samples) / len(self.samples)

class ModelWithB2:
    def __init__(self, H, particle_count, B=0, theta_count=1, phi_count=1):
        # thetas = np.tile(np.linspace(0, np.pi, theta_count), phi_count)
        thetas = np.tile(np.arccos(np.linspace(-1, 1, theta_count)), phi_count)
        # thetas = np.tile(np.arccos(np.linspace(-0.95, 0.95, theta_count)), phi_count)
        phis = np.tile(np.linspace(0, 2*np.pi, phi_count).T, (theta_count, 1)).T.flatten()
        self.thetas = thetas
        self.samples = []
        for i in range(theta_count*phi_count):
            q = np.array([
                np.sin(thetas[i])*np.cos(phis[i]),
                np.sin(thetas[i])*np.sin(phis[i]),
                np.cos(thetas[i])
            ])
            self.samples.append(FixedModelWithB(H, B, q))
    
    def polarisation(self, ts):
        # return sum(m.polarisation(ts)*np.sin(self.thetas[i]) for i, m in enumerate(self.samples)) * np.pi/2 / len(self.samples)
        return sum(m.polarisation(ts) for m in self.samples) / len(self.samples)

class ModelWithBPerpendicular:
    def __init__(self, H, particle_count, B=0, sample_count=1):
        # thetas = np.pi*rng.random(sample_count)
        thetas = np.arccos(1 - 2*np.array([rng.random(samples//10)/10 + i/10 for i in range(10)]).flatten())
        phis = 2*np.pi*rng.random(sample_count)
        self.thetas = thetas
        self.samples = []
        for i in range(sample_count):
            B_axis = np.array([
                np.sin(thetas[i])*np.cos(phis[i]),
                np.sin(thetas[i])*np.sin(phis[i]),
                np.cos(thetas[i])
            ])
            if np.isclose(B_axis[2], 1):
                measurement_axis_1 = np.array([1, 0, 0])
                measurement_axis_2 = np.array([0, 1, 0])
            else:
                measurement_axis_1 = np.array([1, 0, 0]) - np.array([1, 0, 0]).dot(B_axis)*B_axis
                measurement_axis_1 /= norm(measurement_axis_1)
                measurement_axis_2 = np.cross(measurement_axis_1, B_axis)
            
            self.samples.append((
                FixedModelWithB(H, B, B_axis, measurement_axis_1),
                FixedModelWithB(H, B, B_axis, measurement_axis_2)
            ))
    
    def polarisation(self, ts):
        # return sum(
        #     ms[0].polarisation(ts)*np.sin(self.thetas[i]) + ms[1].polarisation(ts)*np.sin(self.thetas[i])
        #     for i, ms in enumerate(self.samples)
        # ) * np.pi/2 / len(self.samples) / 2
        return sum(
            ms[0].polarisation(ts) + ms[1].polarisation(ts)
            for ms in self.samples
        ) / len(self.samples) / 2

compare = False

# DISTANCE = 0.06e-9
DISTANCE = 2.34e-10/2
C = (
    mu_0 *
    FLUORINE_19_GYROMAGNETIC_RATIO *
    MUON_GYROMAGNETIC_RATIO_OVER_2_PI / (2*DISTANCE**3)
)
C2 = (
    mu_0 *
    FLUORINE_19_GYROMAGNETIC_RATIO *
    FLUORINE_19_GYROMAGNETIC_RATIO / (4*np.pi*DISTANCE**3)
)

# B = 0.2
# B = 0.02
# B = 0.002

# B = 220e-4
# B = 0.002
B = 0

H = C*(
    sum(tp(S[i], S[i]) for i in range(3)) - 3*tp(S[2], S[2])
)

# m = Model(C*(
#     sum(tp(S[i], S[i]) for i in range(3)) - 3*tp(S[2], S[2])
# ), 2)
# m = ModelWithB2(C*(
#     sum(tp(S[i], S[i]) for i in range(3)) - 3*tp(S[2], S[2])
# ), 2, B, 30, 1)

if compare:
    m2 = ModelWithB(C*(
        sum(tp(S[i], S[i]) for i in range(3)) - 3*tp(S[2], S[2])
    ), 2, B, 30)
    # m2 = ModelWithBPerpendicular(C*(
    #     sum(tp(S[i], S[i]) for i in range(3)) - 3*tp(S[2], S[2])
    # ), 2, B, 20)

# ts = np.arange(0, 1e-5, 1e-8)


# ts = np.arange(0, 1e-4, 1e-8)
ts = np.arange(0, 2e-5, 1e-8)


# ts = np.arange(0, 26e-6, 1e-8)
# ts = np.arange(0, 1e-3, 1e-8)
# ts = np.arange(0, 1e-2, 1e-5)
# ts = np.arange(0, 1, 1e-3)
# ts = np.arange(0, 1, 1e-6)

samples = 20
# samples = 100

m = ModelWithB(H, 2, B, samples)
# m = ModelWithBPerpendicular(H, 2, B, samples)

# plt.title('Muon spin polarisation')
# plt.plot(ts, m.polarisation(ts))
# plt.plot(ts, ModelWithB2(H, 2, B, samples, 1).polarisation(ts))
# plt.plot(ts, Model(H, 2).polarisation(ts), linestyle='--')
# plt.xlabel('Time / $s^{-1}$')
# plt.ylabel('Polarisation')

plt.figure(figsize=(5, 4))
# plt.title('Muon spin polarisation')
plt.plot(ts*1e6, Model(H, 2).polarisation(ts), color='blue')
# plt.plot(ts, m.polarisation(ts))
plt.plot(ts*1e6, ModelWithB2(H, 2, B, samples, 1).polarisation(ts), color='k', linestyle='--')
plt.xlabel('Time, $t$ / $\mathrm{{\mu}s^{-1}}$')
plt.ylabel('$D_{z}(t)$')
# plt.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\James\Documents\School\Project\figures\Monte.png', dpi=220)

plt.figure()
plt.title('Muon spin polarisation')
plt.plot(ts, m.polarisation(ts) - Model(H, 2).polarisation(ts))
plt.plot(ts, ModelWithB2(H, 2, B, samples, 1).polarisation(ts) - Model(H, 2).polarisation(ts))
plt.xlabel('Time / $s^{-1}$')
plt.ylabel('Error')

plt.figure()
for i in range(11):
    plt.axvline(np.arccos(1-2*i/10))
plt.scatter(m.thetas, [m.samples[i].polarisation(4.4e-6)*np.sin(m.thetas[i]) for i in range(len(m.samples))], marker='x')
m2 = ModelWithB2(H, 2, B, samples, 1)
plt.scatter(m2.thetas, [m2.samples[i].polarisation(4.4e-6)*np.sin(m2.thetas[i]) for i in range(len(m2.samples))], marker='+')

# D = m.polarisation(ts)
# plt.title('Muon spin polarisation')
# plt.plot(ts, D)
# if compare:
#     plt.plot(ts, m2.polarisation(ts))
# plt.xlabel('Time / $s^{-1}$')
# plt.ylabel('Polarisation')

# plt.figure()
# ss = range(10, 100)
# # t = 3.1e-5
# t = 4.5e-6
# H = C*(
#     sum(tp(S[i], S[i]) for i in range(3)) - 3*tp(S[2], S[2])
# )
# plt.title('Convergence')
# p = Model(H, 2).polarisation(t)
# plt.plot(ss, [ModelWithB2(H, 2, B, s, 1).polarisation(t)-p for s in ss])
# plt.plot(ss, [ModelWithB(H, 2, B, s).polarisation(t)-p for s in ss])

plt.show()