import numpy as np
from copy import copy
from numbers import Number

class Subspace:
    def __init__(self, label, parent, I=np.eye(2)):
        self.label = label
        self.parent = parent
        self.I = I

class HilbertSpace:
    def __init__(self):
        self.spaces = []
    def add_subspace(self, label):
        subspace = Subspace(label, self)
        self.spaces.append(subspace)
        return subspace
    def insert_subspace(self, i, label):
        subspace = Subspace(label, self)
        self.spaces.insert(i, subspace)
        return subspace

class SubspaceOp:
    def __init__(self, components, space):
        self.components = components
        self.space = space
    def __array__(self):
        result = self.components.get(self.space.spaces[0].label, self.space.spaces[0].I)
        for s in self.space.spaces[1:]:
            result = np.kron(result, self.components.get(s.label, s.I))
        return result
    def __rmul__(self, other):
        return self.__mul__(other)
    def __mul__(self, other):
        if isinstance(other, Number):
            return self.__class__({l: other*m for (l, m) in self.components.items()}, self.space)
        elif isinstance(other, self.__class__):
            new_components = copy(self.components)
            for (l, m) in other.components.items():
                if l in new_components:
                    new_components[l] = new_components[l] @ m
                else:
                    new_components[l] = m
            return np.array(self.__class__(new_components, self.space))
        else:
            raise NotImplemented
    def dot(self, other):
        if isinstance(other, self.__class__):
            return sum(
                self.__class__(self.components, self.space)
            )

class Op:
    def __init__(self, matrix):
        self.matrix = matrix
    def __call__(self, subspace):
        return SubspaceOp({subspace.label: self.matrix}, subspace.parent)

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
Sz = Op(np.tensordot(sigma, np.array([0, 0, 1]), axes=(0, 0)))

space = HilbertSpace()
muon = space.add_subspace(0)
fluorine = space.add_subspace(1)

print(
    # sum(Op(sigma[i])(muon)*Op(sigma[i])(fluorine) for i in range(3))
    # np.sum(S(muon)*S(fluorine), axis=0)
    # S(muon)*S(fluorine)
    np.sum(np.matmul(S(muon), S(fluorine)), axis=0)
    - 3*Sz(muon)*Sz(fluorine)
)