import numpy as np
from numpy.linalg import eigh
from scipy.constants import hbar, mu_0
from matplotlib import pyplot as plt

FLUORINE_19_GYROMAGNETIC_RATIO = 251.662e6
MUON_GYROMAGNETIC_RATIO_OVER_2_PI = 136e6

def adj(a):
    return a.conj().T

def tp(*args):
    result = args[0]
    for t in args[1:]:
        result = np.kron(result, t)
    return result

I = np.eye(2)
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

DISTANCE = 0.06e-9
C = (
    mu_0 *
    FLUORINE_19_GYROMAGNETIC_RATIO *
    MUON_GYROMAGNETIC_RATIO_OVER_2_PI / (2*DISTANCE**3)
)

H = C*(
    sum(tp(S[i], S[i], I) for i in range(3)) - 3*tp(S[2], S[2], I)
    + sum(tp(S[i], I, S[i]) for i in range(3)) - 3*tp(S[2], I, S[2])
    # + sum(tp(I, S[i], S[i]) for i in range(3)) - 3*tp(I, S[2], S[2]) / 8
)
energies, M = eigh(H)
N = len(energies)

sigma_x = np.array([
    [0, 1],
    [1, 0]
])

sigma_z = np.array([
    [1, 0],
    [0, -1]
])

ts = np.arange(0, 1e-5, 1e-8)

transition_amplitude_matrix = (
    2 * (np.abs(adj(M) @ tp(sigma_x, I, I) @ M)**2)
    + np.abs(adj(M) @ tp(sigma_z, I, I) @ M)**2
) / 3 / 8

frequency_matrix = np.abs(np.tile(energies, (N, 1)).T - np.tile(energies, (N, 1))) / hbar

D = sum(
    transition_amplitude_matrix[i, j] * np.cos(frequency_matrix[i, j]*ts)
    for i in range(N) for j in range(N)
)

rounded_frequencies = np.round(energies / hbar)
frequency_levels = set(rounded_frequencies)

transitions = set()

for i in range(N):
    for j in range(N):
        if transition_amplitude_matrix[i, j] > 1e-8:
            a = rounded_frequencies[i]
            b = rounded_frequencies[j]
            if a > b:
                a, b = b, a
            if a != b:
                transitions.add((a, b))

for f in frequency_levels:
    plt.axhline(f, 0, len(transitions))

x = 0.5
for (a, b) in transitions:
    # plt.arrow(x, a, 0, b-a)
    plt.annotate('', (x, a), (x, b), arrowprops={'arrowstyle': '<->'})
    x += 1

plt.title('Muon spin energy levels')
plt.xlim(0, len(transitions))
plt.ylabel('Frequency / $GHz$')
plt.figure()

cos_amplitudes = {}

for i in range(N):
    for j in range(N):
        cos_amplitudes[frequency_matrix[i, j].round()] = \
            cos_amplitudes.get(frequency_matrix[i, j].round(), 0) + transition_amplitude_matrix[i, j]

for (f, a) in cos_amplitudes.items():
    plt.axvline(f * 1e-6, 0, a)

plt.title('Cosine amplitudes')
plt.xlabel('Frequency / $GHz$')
plt.ylabel('Amplitude')
plt.figure()

plt.title('Muon spin polarisation')
plt.plot(ts, D)
# plt.ylim(-1, 1)
plt.xlabel('Time / $s^{-1}$')
plt.ylabel('Polarisation')
plt.show()