from sympy import I, S, symbols, exp, Sum, cos, simplify, sqrt
from sympy.matrices import Matrix, eye, zeros
from sympy.physics.quantum import TensorProduct, represent, get_basis, hbar, IdentityOperator, Dagger, OuterProduct
from sympy.physics.quantum.spin import Jx, Jy, Jz, JxKet, JzKet, couple

up = JzKet(1/S(2), 1/S(2))
down = JzKet(1/S(2), -1/S(2))

fluorine_density_matrix = (OuterProduct(up, up.dual) + OuterProduct(down, down.dual))/2

muon_state_x = JxKet(1/S(2), 1/S(2))
muon_density_matrix_x = OuterProduct(muon_state_x, muon_state_x.dual)
density_matrix_x = represent(TensorProduct(muon_density_matrix_x, fluorine_density_matrix, fluorine_density_matrix), basis=Jz)

muon_state_z = JzKet(1/S(2), 1/S(2))
muon_density_matrix_z = OuterProduct(muon_state_z, muon_state_z.dual)
density_matrix_z = represent(TensorProduct(muon_density_matrix_z, fluorine_density_matrix, fluorine_density_matrix), basis=Jz)

sigma_x = TensorProduct(represent(2*Jx/hbar, basis=Jz), eye(2), eye(2))
sigma_z = TensorProduct(represent(2*Jz/hbar, basis=Jz), eye(2), eye(2))

H = symbols('C')*(
    TensorProduct(Jx, Jx, eye(2)) + TensorProduct(Jy, Jy, eye(2)) - 2*TensorProduct(Jz, Jz, eye(2)) 
    + TensorProduct(Jx, eye(2), Jx) + TensorProduct(Jy, eye(2), Jy) - 2*TensorProduct(Jz, eye(2), Jz)
)
eigenstuff = represent(H, basis=Jz).eigenvects(simplify=True)

eigenenergies = sum([[e[0]]*e[1] for e in eigenstuff], start=[])
eigenvectors = [v.normalized() for e in eigenstuff for v in e[2]]

for i in range(8):
    print(eigenenergies[i])
    print(eigenvectors[i])
    print()

# t = symbols('t')

# Px = sum([
#     (
#         Dagger(eigenvectors[n]) * density_matrix_x * eigenvectors[m] *
#         Dagger(eigenvectors[m]) * sigma_x * eigenvectors[n] * exp(I*t*(eigenenergies[m]-eigenenergies[n])/hbar)
#     )[0, 0]
#     for n in range(len(eigenvectors)) for m in range(len(eigenvectors))
# ])
# Pz = sum([
#     (
#         Dagger(eigenvectors[n]) * density_matrix_z * eigenvectors[m] *
#         Dagger(eigenvectors[m]) * sigma_z * eigenvectors[n] * exp(I*t*(eigenenergies[m]-eigenenergies[n])/hbar)
#     )[0, 0]
#     for n in range(len(eigenvectors)) for m in range(len(eigenvectors))
# ])

# print(simplify(((2*Px + Pz).doit()/3).rewrite(cos)))