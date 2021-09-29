import numpy as np

def partial_trace2(density_matrix: np.ndarray, index: int) -> np.ndarray:
    """
    Takes an operator density_matrix belonging to the 2**N dimensional Hilbert space
    formed from the tensor product of the 2 dimensional subspaces, 0,...,N.

    Returns the partial trace over all subspaces other than index.
    """
    N = int(np.log2(density_matrix.shape[0]))
    if index != 0:
        density_matrix = np.moveaxis(
            density_matrix.reshape(np.full(2*N, 2)),
            (0, index, N, N+index),
            (index, 0, N+index, N)
        )
    return density_matrix.reshape(2, 2**(N-1), 2, 2**(N-1)).trace(axis1=1, axis2=3)