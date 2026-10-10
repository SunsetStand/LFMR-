"""General electronic integrals × independent truncated oscillator axes."""

import numpy as np
from pyscf import lib

from ._electronic import ElectronicSpace
from .phonon import boson_operators, phonon_configs


class GeneralHamiltonian:
    """Matrix-free action; state order [alpha address,beta address,mode 0,...]."""

    def __init__(self, model, nmax):
        self.model, self.nmax = model, nmax
        self.electronic = ElectronicSpace(model)
        self.shape = self.electronic.shape + (nmax + 1,) * model.nmode
        self.size = int(np.prod(self.shape))
        self.configs = (phonon_configs(model.nmode, nmax) if model.nmode
                        else np.zeros((1, 0), dtype=np.int64))
        self.phonon_energy = self.configs @ model.omega
        self.position = sum(boson_operators(nmax)[:2])

    def __call__(self, psi):
        if np.iscomplexobj(psi):
            raise TypeError("general ED currently requires real state coefficients")
        psi = np.asarray(psi, dtype=np.float64)
        if psi.shape != self.shape:
            raise ValueError(f"psi must have structured shape {self.shape}")
        ci = psi.reshape(*self.electronic.shape, -1)
        result = np.empty_like(ci)
        for k in range(ci.shape[-1]):
            result[..., k] = self.electronic.apply(ci[..., k])
        result += ci * (self.model.ecore + self.phonon_energy)
        result = result.reshape(self.shape)
        for mode, g in enumerate(self.model.coupling):
            electronic = np.empty_like(ci)
            for k in range(ci.shape[-1]):
                electronic[..., k] = self.electronic.apply_one(g, ci[..., k])
            moved = np.moveaxis(electronic.reshape(self.shape), 2 + mode, -1)
            acted = np.einsum("ij,...j->...i", self.position, moved)
            result += np.moveaxis(acted, -1, 2 + mode)
        return result

    def diagonal(self):
        diagonal = self.electronic.diagonal().reshape(*self.electronic.shape, 1)
        return (diagonal + self.model.ecore + self.phonon_energy).reshape(-1)


def solve(model, nmax, *, tol, tol_residual, max_cycle, max_space, lindep):
    ham = GeneralHamiltonian(model, nmax)
    diagonal = ham.diagonal()
    if not sum(model.nelec):
        # No electron operator survives, and positive-frequency bare
        # oscillators have an exact vacuum ground state.
        vacuum = np.zeros(ham.shape)
        vacuum.flat[0] = 1.
        return model.ecore, vacuum, 0.
    # A diagonal-minimum determinant alone can miss a lower, disconnected
    # electronic/symmetry block. Even adding it as a second seed may allow
    # premature convergence on that exact eigenvector. Start with full support.
    guess = np.random.default_rng(417).normal(size=ham.size)
    guess /= np.linalg.norm(guess)

    def hop(vector):
        return ham(vector.reshape(ham.shape)).ravel()

    _, energies, vectors = lib.davidson1(
        lambda xs: [hop(x) for x in xs], [guess],
        lib.make_diag_precond(diagonal, level_shift=.05),
        tol=tol, tol_residual=tol_residual, max_cycle=max_cycle,
        max_space=max_space, lindep=lindep, verbose=0,
    )
    energy, vector = float(energies[0]), vectors[0]
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError("Davidson returned an invalid state; use a larger lindep threshold")
    vector = vector / norm
    residual = float(np.linalg.norm(hop(vector) - energy * vector))
    return energy, vector.reshape(ham.shape), residual
