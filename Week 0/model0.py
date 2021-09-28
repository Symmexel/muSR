import numpy as np
from numpy.linalg import eigh
from numpy.random import default_rng
from scipy.constants import hbar, mu_0
from matplotlib import pyplot as plt

FLUORINE_19_GYROMAGNETIC_RATIO = 250e6
MUON_GYROMAGNETIC_RATIO_TIMES_2_PI = 136e6
DISTANCE = 0.06e-9
C = (
    mu_0 *
    FLUORINE_19_GYROMAGNETIC_RATIO *
    MUON_GYROMAGNETIC_RATIO_TIMES_2_PI / (2*DISTANCE**3)
)
MUON_MEAN_LIFETIME = 2.2e-6
MUON_COUNT = 100

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
projection_up = np.outer(up, up.conj())

rng = default_rng()

def random_lifetime(size=None):
    return -MUON_MEAN_LIFETIME*np.log(rng.random(size))

def measure_is_spin_forward(state):
    return rng.random() < state.conj() @ np.kron(projection_up, I) @ state

n = np.array([1, 0, 0])
Sn = np.tensordot(S, n, axes=(0, 0))
H = C*(sum(np.kron(S[i], S[i]) for i in range(3)) - 3*np.kron(Sn, Sn))
vals, M = eigh(H)

def U(t):
    return M @ np.diag(np.exp(vals*(-1j*t/hbar))) @ M.conj().T

muon_state = down
fluorine_state = up
state = np.kron(muon_state, fluorine_state)

forward_events = []
backward_events = []

for lifetime in random_lifetime(MUON_COUNT):
    if measure_is_spin_forward(U(lifetime) @ state):
        forward_events.append(lifetime)
    else:
        backward_events.append(lifetime)

forward_events.sort()
backward_events.sort()

def cumulative_forward_events(t):
    count = 0
    while count < len(forward_events) and forward_events[count] < t:
        count += 1
    return count

def cumulative_backward_events(t):
    count = 0
    while count < len(backward_events) and backward_events[count] < t:
        count += 1
    return count

def calculate_rates(bin_count, step, sorted_events):
    result = np.ndarray(bin_count)
    i = 0
    j = 0
    while i < len(sorted_events) and j < bin_count:
        count = 0
        while i < len(sorted_events) and sorted_events[i] < (j+1) * step:
            count += 1
            i += 1
        result[j] = count/step
        j += 1
    return result

step = 1e-8
ts = np.arange(0, 1e-5, step)

N_backward = calculate_rates(len(ts), step, backward_events)
N_forward = calculate_rates(len(ts), step, forward_events)
A = (N_backward-N_forward) / (N_backward+N_forward)
G = A/np.max(A)

sigma_z = np.array([
    [1, 0],
    [0, -1]
])

def D(state):
    return state.conj() @ np.kron(sigma_z, I) @ state

# plt.plot(ts, [cumulative_backward_events(t) for t in ts], label='Backwards')
# plt.plot(ts, [cumulative_forward_events(t) for t in ts], label='Forwards')
# plt.legend(title='Cumulative Events')
# plt.plot(ts, N_backward, label='$N_{backward}$')
# plt.plot(ts, N_forward, label='$N_{forward}$')
# plt.plot(ts, N_backward+N_forward, label='$N_{backward}+N_{forward}$')
# plt.legend()
# plt.plot(ts, G)
plt.plot(ts, np.array([-D(U(t) @ state) for t in ts]).real)
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.show()