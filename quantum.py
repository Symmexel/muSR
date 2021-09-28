import numpy as np
from numbers import Number

class SubspaceOp:
    def __init__(self, matrix, index, n):
        self.matrix = matrix
        self.index = index
        self.n = n
    def __array__(self):
        result = 1
        for i in range(self.n):
            if i == self.index:
                result = np.kron(result, self.matrix)
            else:
                result = np.kron(result, np.eye(2))
        return result
    def __rmul__(self, other):
        return self.__mul__(other)
    def __mul__(self, other):
        if isinstance(other, Number):
            return self.__class__(other*self.matrix, self.index, self.n)
        elif isinstance(other, self.__class__):
            if self.index == other.index:
                return np.array(self.__class__(self.matrix @ other.matrix, self.index, self.n))
            else:
                result = 1
                for i in range(max(self.n, other.n)):
                    if i == self.index:
                        result = np.kron(result, self.matrix)
                    elif i == other.index:
                        result = np.kron(result, other.matrix)
                    else:
                        result = np.kron(result, np.eye(2))
                return result
        else:
            raise NotImplemented
    def dot(self, other):
        if isinstance(other, self.__class__):
            return sum(
                self.__class__(
                    self.matrix[i], self.index, self.n
                ) * self.__class__(
                    other.matrix[i], other.index, other.n
                ) for i in range(3)
            )
        else:
            raise NotImplemented

class Op:
    def __init__(self, matrix):
        self.matrix = matrix
    def __call__(self, subspace):
        return SubspaceOp(self.matrix, subspace[0], subspace[1])
    def dot(self, other):
        return Op(np.tensordot(self.matrix, other, axes=(0, 0)))

def reduce_trace(density_matrix, index):
    N = int(np.log2(density_matrix.shape[0]))
    if index != 0:
        density_matrix = np.moveaxis(
            density_matrix.reshape(np.full(2*N, 2)),
            (0, index, N, N+index),
            (index, 0, N+index, N)
        )
    return density_matrix.reshape(2, 2**(N-1), 2, 2**(N-1)).trace(axis1=1, axis2=3)

if __name__ == "__main__":
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

    S = Op(sigma)
    Sz = S.dot(np.array([0, 0, 1]))
    # Sz = Op(np.tensordot(sigma, np.array([0, 0, 1]), axes=(0, 0)))

    muon = (0, 2)
    fluorine = (1, 2)

    print(
        # sum(Op(sigma[i])(muon)*Op(sigma[i])(fluorine) for i in range(3))
        # np.sum(np.matmul(S(muon), S(fluorine)), axis=0)
        S(muon).dot(S(fluorine))
        - 3*Sz(muon)*Sz(fluorine)
    )

    density_matrix = 1
    for dm in [
        np.array([[1, 0], [0, 0]]),
        np.eye(2)/2,
        np.array([[0, 0], [0, 1]]),
        np.array([[1, 0], [0, 0]])
    ]:
        density_matrix = np.kron(density_matrix, dm)
    
    for i in range(0, 4):
        print(reduce_trace(density_matrix, i))