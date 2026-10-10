"""Matrix-free exact diagonalization in the site/Fock basis."""

import numpy as np
from pyscf import lib

from ._driver import Calculation, require_reference
from .basis import make_electron_basis, make_shape
from .model import ElectronPhononModel
from .phonon import boson_operators, phonon_configs


def contract_1e(tmat: np.ndarray, psi_site: np.ndarray, L: int, nelec: tuple[int, int], Nmax: int):
    psi_shape = make_shape(L, nelec, Nmax)
    if not isinstance(tmat, np.ndarray) or tmat.shape != (L, L):
        raise ValueError("tmat must be a NumPy array with shape (L, L)")

    if not isinstance(psi_site, np.ndarray) or psi_site.shape != psi_shape:
        raise ValueError("psi_site has an incompatible shape")
    new_psi = np.zeros(psi_shape)
    neleca, nelecb = nelec
    _, links_a = make_electron_basis(L, neleca)
    _, links_b = make_electron_basis(L, nelecb)
    for str0, tab in enumerate(links_a):
        for a, i, str1, sign in tab:
            new_psi[str1] += sign * tmat[a, i] * psi_site[str0]
    for str0, tab in enumerate(links_b):
        for a, i, str1, sign in tab:
            new_psi[:, str1] += sign * tmat[a, i] * psi_site[:, str0]
    return new_psi


def contract_2e_hubbard(U: float, psi_site: np.ndarray, L: int, nelec: tuple[int, int], Nmax: int):
    psi_shape = make_shape(L, nelec, Nmax)
    if not isinstance(psi_site, np.ndarray) or psi_site.shape != psi_shape:
        raise ValueError("psi_site has an incompatible shape")
    if not isinstance(U, (int, float, np.integer, np.floating)) or not np.isfinite(U):
        raise ValueError("U must be a finite real number")
    new_psi = np.zeros(psi_shape)
    neleca, nelecb = nelec
    strs_a, _ = make_electron_basis(L, neleca)
    strs_b, _ = make_electron_basis(L, nelecb)
    for ia, str_a in enumerate(strs_a):
        for ib, str_b in enumerate(strs_b):
            new_psi[ia, ib] = U * int(str_a & str_b).bit_count() * psi_site[ia, ib]
    return new_psi


def contract_pp(hpp: np.ndarray, psi_site: np.ndarray, L: int, nelec: tuple[int, int], Nmax: int):
    psi_shape = make_shape(L, nelec, Nmax)
    if not isinstance(psi_site, np.ndarray) or psi_site.shape != psi_shape:
        raise ValueError("psi_site has an incompatible shape")
    if not isinstance(hpp, np.ndarray) or not hpp.shape == (L, L):
        raise ValueError("hpp has an incompatible shape")
    if (
        not np.issubdtype(hpp.dtype, np.number)
        or np.iscomplexobj(hpp)
        or not np.all(np.isfinite(hpp))
    ):
        raise ValueError("hpp must contain finite real numbers")
    hpp_off_diagonal = hpp - np.diag(np.diag(hpp))
    if not np.allclose(hpp_off_diagonal, 0.0):
        raise ValueError("hpp must be diagonal")
    omega = np.diag(hpp)
    configs = phonon_configs(L, Nmax)
    d = Nmax + 1
    phonon_energies = configs @ omega
    energy_tensor = phonon_energies.reshape((1, 1) + (d,) * L)
    new_psi = energy_tensor * psi_site
    return new_psi


def contract_ep_paper(g: float, psi_site: np.ndarray, L: int, nelec: tuple[int, int], Nmax: int):
    if (
        isinstance(g, (bool, np.bool_))
        or not isinstance(g, (int, float, np.integer, np.floating))
        or not np.isfinite(g)
    ):
        raise ValueError("g must be finite real scalar")
    psi_shape = make_shape(L, nelec, Nmax)
    if psi_site.shape != psi_shape:
        raise ValueError("psi_site has the wrong shape")
    nelec_a, nelec_b = nelec
    strs_a, _ = make_electron_basis(L, nelec_a)
    strs_b, _ = make_electron_basis(L, nelec_b)
    num_a = len(strs_a)
    num_b = len(strs_b)
    occupation_site = np.zeros((num_a, num_b, L))
    for ia in range(num_a):
        for ib in range(num_b):
            for site in range(L):
                occupation_site[ia, ib, site] = bool(strs_a[ia] & (1 << site)) + bool(
                    strs_b[ib] & (1 << site)
                )
    b, bdag, _ = boson_operators(Nmax)
    x_local = b + bdag
    out_dtype = np.result_type(psi_site.dtype, x_local.dtype, g)
    new_psi = np.zeros(psi_shape, dtype=out_dtype)
    for site in range(L):
        occupation = occupation_site[:, :, site]
        occupation_tensor = occupation.reshape((num_a, num_b) + (1,) * L)
        weighted_psi = occupation_tensor * psi_site
        phonon_axis = 2 + site
        weighted_moved = np.moveaxis(weighted_psi, phonon_axis, -1)
        acted_moved = np.einsum("mn,...n->...m", x_local, weighted_moved)
        acted = np.moveaxis(acted_moved, -1, phonon_axis)
        new_psi += g * acted
    return new_psi


def contract_ep_centered(g: float, psi_site: np.ndarray, L: int, nelec: tuple[int, int], Nmax: int):
    if (
        isinstance(g, (bool, np.bool_))
        or not isinstance(g, (int, float, np.integer, np.floating))
        or not np.isfinite(g)
    ):
        raise ValueError("g must be finite real scalar")
    psi_shape = make_shape(L, nelec, Nmax)
    if psi_site.shape != psi_shape:
        raise ValueError("psi_site has the wrong shape")
    nelec_a, nelec_b = nelec
    strs_a, _ = make_electron_basis(L, nelec_a)
    strs_b, _ = make_electron_basis(L, nelec_b)
    num_a = len(strs_a)
    num_b = len(strs_b)
    occupation_site = np.zeros((num_a, num_b, L))
    average_occupation = (nelec_a + nelec_b) / L
    for ia in range(num_a):
        for ib in range(num_b):
            for site in range(L):
                occupation_site[ia, ib, site] = bool(strs_a[ia] & (1 << site)) + bool(
                    strs_b[ib] & (1 << site)
                )
    ocp_centered = occupation_site - average_occupation
    b, bdag, _ = boson_operators(Nmax)
    x_local = b + bdag
    out_dtype = np.result_type(psi_site.dtype, x_local.dtype, g)
    new_psi = np.zeros(psi_shape, dtype=out_dtype)
    for site in range(L):
        occupation = ocp_centered[:, :, site]
        occupation_tensor = occupation.reshape((num_a, num_b) + (1,) * L)
        weighted_psi = occupation_tensor * psi_site
        phonon_axis = 2 + site
        weighted_moved = np.moveaxis(weighted_psi, phonon_axis, -1)
        acted_moved = np.einsum("mn,...n->...m", x_local, weighted_moved)
        acted = np.moveaxis(acted_moved, -1, phonon_axis)
        new_psi += g * acted
    return new_psi


def contract_all(
    tmat: np.ndarray,
    U: float,
    g: float,
    hpp: np.ndarray,
    psi_site: np.ndarray,
    L: int,
    nelec: tuple[int, int],
    Nmax: int,
    coupling_convention="paper",
):
    psi_shape = make_shape(L, nelec, Nmax)
    if not isinstance(psi_site, np.ndarray) or psi_site.shape != psi_shape:
        raise ValueError("psi_site has an incompatible shape")
    if coupling_convention not in ["paper", "centered"]:
        raise ValueError("coupling_convention must be either 'paper' or 'centered'")
    new_psi = contract_1e(tmat, psi_site, L, nelec, Nmax)
    new_psi += contract_2e_hubbard(U, psi_site, L, nelec, Nmax)
    new_psi += contract_pp(hpp, psi_site, L, nelec, Nmax)
    if coupling_convention == "paper":
        new_psi += contract_ep_paper(g, psi_site, L, nelec, Nmax)
    else:
        new_psi += contract_ep_centered(g, psi_site, L, nelec, Nmax)
    return new_psi


def make_hdiag(
    tmat: np.ndarray, U: float, g: float, hpp: np.ndarray, L: int, nelec: tuple[int, int], Nmax: int
):
    if not isinstance(tmat, np.ndarray) or tmat.shape != (L, L):
        raise ValueError("tmat must be a NumPy array with shape (L, L)")
    if not isinstance(hpp, np.ndarray) or not hpp.shape == (L, L):
        raise ValueError("hpp has an incompatible shape")
    if (
        not np.issubdtype(hpp.dtype, np.number)
        or np.iscomplexobj(hpp)
        or not np.all(np.isfinite(hpp))
    ):
        raise ValueError("hpp must contain finite real numbers")
    hpp_off_diagonal = hpp - np.diag(np.diag(hpp))
    if not np.allclose(hpp_off_diagonal, 0.0):
        raise ValueError("hpp must be diagonal")
    omega = np.diag(hpp)
    configs = phonon_configs(L, Nmax)
    d = Nmax + 1
    phonon_energies = configs @ omega
    strs_a, _ = make_electron_basis(L, nelec[0])
    strs_b, _ = make_electron_basis(L, nelec[1])
    num_a = len(strs_a)
    num_b = len(strs_b)
    occupation_site = np.zeros((num_a, num_b, L))
    for ia in range(num_a):
        for ib in range(num_b):
            for site in range(L):
                occupation_site[ia, ib, site] = bool(strs_a[ia] & (1 << site)) + bool(
                    strs_b[ib] & (1 << site)
                )
    double_occ = np.zeros((num_a, num_b))
    for ia, str_a in enumerate(strs_a):
        for ib, str_b in enumerate(strs_b):
            double_occ[ia, ib] = int(str_a & str_b).bit_count()
    t_diag = np.diag(tmat)
    Et = np.einsum("abs,s->ab", occupation_site, t_diag)
    hdiag_tensor = (
        U * double_occ.reshape((num_a, num_b) + (1,) * L)
        + phonon_energies.reshape((1, 1) + (d,) * L)
        + Et.reshape((num_a, num_b) + (1,) * L)
    )
    return hdiag_tensor.reshape(-1)


def kernel(
    tmat: np.ndarray,
    U: float,
    g: float,
    hpp: np.ndarray,
    L: int,
    nelec: tuple[int, int],
    Nmax: int,
    tol: float = 1e-12,
    max_cycle: int = 100,
    max_space: int = 20,
    verbose: int = 0,
    tol_residual: float | None = None,
    lindep: float = 1e-14,
    coupling_convention: str = "paper",
):
    psi_shape = make_shape(L, nelec, Nmax)
    D = int(np.prod(psi_shape))

    def hop(vector):
        psi_site = vector.reshape(psi_shape)
        hpsi_site = contract_all(tmat, U, g, hpp, psi_site, L, nelec, Nmax, coupling_convention)
        return hpsi_site.reshape(-1)

    hdiag = make_hdiag(tmat, U, g, hpp, L, nelec, Nmax)
    guess_address = int(np.argmin(hdiag))
    x0 = np.zeros(D, dtype=np.float64)
    x0[guess_address] = 1.0
    precond = lib.make_diag_precond(hdiag)

    def hop_many(vectors):
        return [hop(vector) for vector in vectors]

    _, energies, vectors = lib.davidson1(
        hop_many,
        [x0],
        precond,
        tol=tol,
        tol_residual=tol_residual,
        lindep=lindep,
        max_cycle=max_cycle,
        max_space=max_space,
        verbose=verbose,
    )
    energy = energies[0]
    vec = vectors[0]
    residual = hop(vec) - energy * vec
    residual_norm = np.linalg.norm(residual)
    vec = vec.reshape(psi_shape)
    return float(energy), vec, float(residual_norm)


class ExactED(Calculation):
    """Davidson at a fixed per-mode Nmax; residual is independently evaluated.

    converged describes this truncated eigenproblem, not cutoff convergence.
    coupling_convention='centered' solves a different finite-cutoff problem;
    energy is returned in that convention without an automatic scalar shift.
    """

    def __init__(
        self,
        model,
        nmax,
        *,
        tol=1e-12,
        tol_residual=1e-10,
        max_cycle=100,
        max_space=20,
        lindep=1e-14,
        coupling_convention="paper",
    ):
        super().__init__(model)
        if (
            isinstance(nmax, (bool, np.bool_))
            or not isinstance(nmax, (int, np.integer))
            or nmax < 0
        ):
            raise ValueError("nmax must be a nonnegative integer")
        if coupling_convention not in ("paper", "centered"):
            raise ValueError("coupling_convention must be paper or centered")
        if isinstance(model, ElectronPhononModel) and coupling_convention != "paper":
            raise ValueError("general integral models use only their supplied uncentered coupling")
        for name, value in [("tol", tol), ("tol_residual", tol_residual), ("lindep", lindep)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.nmax, self.tol, self.tol_residual = int(nmax), tol, tol_residual
        self.max_cycle, self.max_space, self.lindep = max_cycle, max_space, lindep
        self.coupling_convention = coupling_convention

    def kernel(self):
        self._reset()
        m = self.model
        if isinstance(m, ElectronPhononModel):
            from .general_ed import solve

            energy, coeff, residual = solve(
                m, self.nmax, tol=self.tol, tol_residual=self.tol_residual,
                max_cycle=self.max_cycle, max_space=self.max_space, lindep=self.lindep,
            )
            result = {
                "energy": energy, "coeff": coeff, "residual": residual,
                "nmax": self.nmax, "dimension": coeff.size,
                "norb": m.norb, "nmode": m.nmode, "nelec": m.nelec, "ecore": m.ecore,
                "solver_converged": residual < self.tol_residual,
                "cutoff_converged": False, "coupling_convention": "paper",
            }
            return self._finish(result, energy, result["solver_converged"])
        energy, coeff, residual = kernel(
            m.tmat,
            m.U,
            m.g,
            m.omega * np.eye(m.L),
            m.L,
            m.nelec,
            self.nmax,
            tol=self.tol,
            tol_residual=self.tol_residual,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            lindep=self.lindep,
            coupling_convention=self.coupling_convention,
        )
        result = {
            "energy": energy,
            "coeff": coeff,
            "residual": residual,
            "nmax": self.nmax,
            "dimension": coeff.size,
            "solver_converged": residual < self.tol_residual,
            "cutoff_converged": False,
            "coupling_convention": self.coupling_convention,
        }
        return self._finish(result, energy, result["solver_converged"])

    def site_density(self):
        require_reference(self)
        from .observables import exact_site_density

        norb = self.model.norb if isinstance(self.model, ElectronPhononModel) else self.model.L
        return exact_site_density(self.result["coeff"], norb, self.model.nelec)
