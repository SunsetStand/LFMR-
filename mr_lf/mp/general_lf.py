"""Finite electronic-space LF-MP2 with all electronic ranks up to doubles.

Analytic infinite-oscillator vacuum-to-number matrix elements avoid truncating
the LF displacement itself. PySCF handles electronic operators and CI basis
changes. The dense electronic oracle is limited to 256 determinants in v1.
"""

import math

import numpy as np
from pyscf.fci import addons

from .._electronic import ElectronicSpace
from ..phonon import lf_total_phonon_configurations
from ..scf.general_lf import fixed_reference, validate_frame


def vacuum_source_matrix(space, ham, coupling, lam, shift, config):
    """<I,config|U† H U|J,0> in the fixed orthonormal electronic basis.

    eta_I=z-lambda*n(I), delta_IJ=eta_J-eta_I.
    <n|D(delta)|0>=exp(-delta²/2)*delta^n/sqrt(n!).
    <n|D†(eta_I)(b_x+b_x†)D(eta_J)|0>
      =(eta_Ix+eta_Jx)*A_n + sqrt(n_x)*A_(n-e_x).
    """
    model = space.model
    eta = shift[None, :] - space.density @ lam.T
    delta = eta[None, :, :] - eta[:, None, :]
    gaussian = np.exp(-.5*np.sum(delta**2, axis=-1))

    def amplitude(occupations):
        value = gaussian.copy()
        for x, count in enumerate(occupations):
            for k in range(1, int(count) + 1):
                value *= delta[:, :, x] / math.sqrt(k)
        return value

    a = amplitude(config)
    matrix = ham * a
    for x, g in enumerate(coupling):
        position = (eta[:, x, None] + eta[None, :, x]) * a
        if config[x]:
            lower = config.copy()
            lower[x] -= 1
            position += math.sqrt(int(config[x])) * amplitude(lower)
        matrix += g * position
    total = int(np.sum(config))
    if total == 0:
        diagonal = model.ecore + eta**2 @ model.omega
        matrix[np.diag_indices(space.size)] += diagonal
    elif total == 1:
        x = int(np.argmax(config))
        matrix[np.diag_indices(space.size)] += model.omega[x] * eta[:, x]
    return matrix


def correction(hf, *, max_total, max_excited_modes, denominator_tol):
    m = hf.model
    for name, value in (("max_total", max_total), ("max_excited_modes", max_excited_modes)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not np.isfinite(denominator_tol) or denominator_tol <= 0:
        raise ValueError("denominator_tol must be finite and positive")
    # Reject before determinant/link enumeration, which itself grows
    # combinatorially for molecular bases outside this limited implementation.
    dimension = math.prod(math.comb(m.norb, count) for count in m.nelec)
    if dimension > 256:
        raise NotImplementedError("finite-space LF-MP2 requires <= 256 electronic determinants")
    space = ElectronicSpace(m)
    ham, coupling = space.dense()
    lam, shift = validate_frame(m, hf.lam, hf.shift)
    reference = space.reference_ci(hf.mo_coeff).ravel()
    energy, _, fock = fixed_reference(m, hf.mo_coeff, lam, shift)
    if abs(energy - hf.e_tot) > 1e-7:
        raise ValueError("MP2 reference does not reproduce supplied LF-HF energy")
    fock_mo = np.array([c.T @ f @ c for c, f in zip(hf.mo_coeff, fock)])
    if np.max(abs(fock_mo - np.array([np.diag(e) for e in hf.mo_energy])), initial=0.) > 1e-6:
        raise ValueError("LF-MP2 requires canonical, stationary Fock orbitals")

    configs = np.zeros((1, m.nmode), dtype=np.int64)
    if m.nmode and max_total:
        nonzero = lf_total_phonon_configurations(m.nmode, max_total)
        nonzero = nonzero[np.count_nonzero(nonzero, axis=1) <= max_excited_modes]
        configs = np.concatenate((configs, nonzero), axis=0)
    ranks, gaps = [], []
    for occ, count, eps in zip(space.occupations, m.nelec, hf.mo_energy):
        ranks.append(count - occ[:, :count].sum(axis=1))
        gaps.append(occ @ eps - eps[:count].sum())
    rank = (ranks[0][:, None] + ranks[1][None, :]).ravel().astype(int)
    gap = (gaps[0][:, None] + gaps[1][None, :]).ravel()
    values = {name: 0. for name in ("phonon_pure_correction", "phonon_single_correction",
                                  "phonon_double_correction", "zero_single_correction",
                                  "zero_double_correction")}
    minimum = float("inf")
    reference_error = 0.
    higher_rank_error = 0.
    for config in configs:
        matrix = vacuum_source_matrix(space, ham, coupling, lam, shift, config)
        source_ao = (matrix @ reference).reshape(space.shape)
        source = addons.transform_ci_for_orbital_rotation(
            source_ao, m.norb, m.nelec, hf.mo_coeff
        ).ravel()
        nonvacuum = bool(np.any(config))
        if not nonvacuum:
            reference_error = abs(source[0] - hf.e_tot)
        higher_rank_error = max(higher_rank_error, float(np.max(abs(source[rank > 2]), initial=0.)))
        for r in (0, 1, 2) if nonvacuum else (1, 2):
            selected = rank == r
            denominator = gap[selected] + np.dot(m.omega, config)
            if denominator.size:
                minimum = min(minimum, float(np.min(denominator)))
                if np.any(denominator <= denominator_tol):
                    raise ValueError("nonpositive or near-zero LF-MP2 excitation denominator")
            name = (("phonon_pure_correction", "phonon_single_correction", "phonon_double_correction")[r]
                    if nonvacuum else ("zero_single_correction" if r == 1 else "zero_double_correction"))
            values[name] -= float(np.sum(source[selected]**2 / denominator))
    if reference_error > 1e-8 or higher_rank_error > 1e-8:
        raise ValueError("LF-MP2 vacuum reference or electronic excitation-rank check failed")
    e_corr = sum(values.values())
    return {
        **values, "hf_energy": energy, "mp2_correction": e_corr,
        "total_energy": energy + e_corr, "occupations": configs[1:],
        "max_total": int(max_total), "max_excited_modes": int(max_excited_modes),
        "minimum_denominator": minimum, "reference_energy_error": reference_error,
        "higher_rank_source_error": higher_rank_error, "electronic_dimension": space.size,
        "partition": "unrestricted canonical LF Fock", "cutoff_converged": False,
    }
