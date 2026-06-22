import numpy as np
import matplotlib.pyplot as plt
from skfem import (
    Basis, BilinearForm, ElementQuad0, ElementQuad1, LinearForm,
    MeshQuad, asm, condense, solve
)
from skfem.helpers import dot, grad

NX, NY = 32, 16
LENGTH, HEIGHT = 2.0, 1.0
PENALTY, E0, E_MIN = 3.0, 1.0, 1e-3


def random_density(seed=0, volume_fraction=0.4):
    rng = np.random.default_rng(seed)
    
    rho = rng.uniform(0.2, 1.0, size=(NY, NX))
    rho *= volume_fraction / rho.mean()
    rho[:, 0] = 1.0  # fixed support
    
    return np.clip(rho, 0.2, 1.0)


def solve_stress(rho):
    rho = np.asarray(rho).reshape(NY, NX)

    mesh = MeshQuad.init_tensor(
        np.linspace(0, LENGTH, NX + 1),
        np.linspace(0, HEIGHT, NY + 1),
    )
    mesh = mesh.with_boundaries({
        "left": lambda x: np.isclose(x[0], 0.0),
    })

    basis = Basis(mesh, ElementQuad1())
    basis0 = basis.with_element(ElementQuad0())

    @BilinearForm
    def stiffness(u, v, w):
        return w.k * dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        return np.where(w.x[0] > 0.95 * LENGTH, 1.0, 0.0) * v

    k = E_MIN + (rho.ravel() ** PENALTY) * (E0 - E_MIN)
    K = asm(stiffness, basis, k=basis0.interpolate(k))
    f = asm(load, basis)
    u = solve(*condense(K, f, D=basis.get_dofs("left")))

    xc = (mesh.p[0, mesh.t[0]] + mesh.p[0, mesh.t[2]]) / 2
    yc = (mesh.p[1, mesh.t[0]] + mesh.p[1, mesh.t[2]]) / 2
    response = basis0.project(basis.interpolate(u))
    return response[np.lexsort((xc, yc))].reshape(NY, NX)


def plot_result(rho, stress):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    extent = (0, LENGTH, 0, HEIGHT)
    axes[0].imshow(rho, origin="lower", extent=extent, cmap="Greys")
    axes[0].set_title("density")
    im = axes[1].imshow(stress, origin="lower", extent=extent, cmap="magma")
    axes[1].set_title("response")
    plt.colorbar(im, ax=axes[1])
    fig.tight_layout()
    return fig


def save_sample(rho, stress, path):
    np.savez(path, rho=rho, stress=stress)
