import numpy as np
from numpy.linalg import eigh
from scipy.constants import hbar, mu_0
from matplotlib import pyplot as plt

def adj(a):
    return a.conj().T

def tp(*args):
    result = args[0]
    for t in args[1:]:
        result = np.kron(result, t)
    return result

FLUORINE_19_GYROMAGNETIC_RATIO = 250e6
MUON_GYROMAGNETIC_RATIO_TIMES_2_PI = 136e6
DISTANCE = 0.06e-9
C = (
    mu_0 *
    FLUORINE_19_GYROMAGNETIC_RATIO *
    MUON_GYROMAGNETIC_RATIO_TIMES_2_PI / (2*DISTANCE**3)
)

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

up = np.array([1, 0])
down = np.array([0, 1])

H = C*(
    sum(tp(S[i], S[i], I) for i in range(3)) - 3*tp(S[2], S[2], I)
    + sum(tp(S[i], I, S[i]) for i in range(3)) - 3*tp(S[2], I, S[2])
)
vals, M = eigh(H)

def U(t):
    return M @ np.diag(np.exp(vals*(-1j*t/hbar))) @ adj(M)

def evolve(density_matrix, t):
    u = U(t)
    return u @ density_matrix @ adj(u)

muon_state = (up+down)/np.sqrt(2)
muon_density_matrix = np.outer(muon_state, muon_state.conj())
fluorine_density_matrix = np.array([
    [0.5, 0],
    [0, 0.5]
])
density_matrix = tp(muon_density_matrix, fluorine_density_matrix, fluorine_density_matrix)

sigma_x = np.array([
    [0, 1],
    [1, 0]
])

sigma_z = np.array([
    [1, 0],
    [0, -1]
])

def D(density_matrix, t):
    return np.trace(evolve(density_matrix, t) @ tp(sigma_x, I, I))

ts = np.arange(0, 1e-5, 1e-8)
plt.plot(ts, np.array([D(density_matrix, t) for t in ts]).real)

# Model 2
energies = vals
N = len(energies)
transition_amplitude_matrix = np.abs(adj(M) @ tp(sigma_x, I, I) @ M)**2
frequency_matrix = (np.tile(energies, (N, 1)) - np.tile(energies, (N, 1)).T) / hbar
D = sum(
    transition_amplitude_matrix[i, j] * np.cos(frequency_matrix[i, j]*ts)
    for i in range(N) for j in range(N)
) / 8
plt.plot(ts, D)

# def f(t):
#     return -np.sqrt(3)*np.sin(3*hbar*C*t/2)*np.sin(np.sqrt(3)*hbar*C*t/2)/6 + np.cos(3*hbar*C*t/2)*np.cos(np.sqrt(3)*hbar*C*t/2)/2 + np.cos(np.sqrt(3)*hbar*C*t)/12 + 5/12
# plt.plot(ts, f(ts))

# plt.plot(ts, [(np.cos(3*C*hbar*t/2)+np.cos(C*hbar*t/2))/2 for t in ts])
plt.title('Muon spin perpendicular to bond')
# plt.plot(ts, [0.5 + 0.5*np.cos(C*hbar*t) for t in ts])
# plt.title('Muon spin parallel to bond')
plt.xlabel('Time / $s^{-1}$')
plt.ylabel('Polarisation')
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.show()