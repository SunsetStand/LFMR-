"""Finite electronic/photon symmetry representations for molecular cavities.

All actions are active ket actions. A photon coherent amplitude transforms as
``eta -> Q(g) @ eta`` in the same operation order as the electronic orbitals.
Only orthogonal mode actions are represented; coherent photon translations
needed by some charged, off-center dipole conventions are not included.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .model import ElectronPhononModel, _real_array
from .molecular import MolecularPF
from .point_group import PointGroupData, _backend, _group_data, _owned, molecular_point_group


@dataclass(frozen=True)
class JointSymmetry:
    """Validated group with orbital ``(G,n,n)`` and photon ``(G,M,M)`` actions.

    Construct with :func:`joint_symmetry` or :func:`molecular_cavity_symmetry`
    to validate against a Hamiltonian. Factory results own read-only float64
    arrays; group irreps use the same operation order as both actions.
    """

    group: PointGroupData
    orbital_actions: np.ndarray
    photon_actions: np.ndarray


def _tolerance(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError("tolerance must be a finite positive real scalar")
    return float(value)


def _covariance_error(model, u, q, tolerance):
    """Return the first failed orthogonality/Hamiltonian condition, or None."""
    def close(a, b):
        return np.allclose(a, b, atol=tolerance, rtol=0)
    if not close(u.T @ u, np.eye(model.norb)):
        return "orbital actions must be orthogonal"
    if not close(q.T @ q, np.eye(model.nmode)):
        return "photon actions must be orthogonal"
    frequencies = np.diag(model.omega)
    if not close(q @ frequencies, frequencies @ q):
        return "photon actions must commute with mode frequencies"
    if not close(u @ model.h1e @ u.T, model.h1e):
        return "one-electron Hamiltonian is not invariant"
    eri = np.einsum("pa,qb,rc,sd,abcd->pqrs", u, u, u, u, model.eri, optimize=True)
    if not close(eri, model.eri):
        return "two-electron Hamiltonian is not invariant"
    transformed = u @ model.coupling @ u.T
    required = np.einsum("yx,ypq->xpq", q, model.coupling)
    if not close(transformed, required):
        return "electron-photon coupling is not jointly invariant"
    return None


def joint_symmetry(model, group, orbital_actions, photon_actions, *, tolerance=1e-9):
    r"""Validate an explicit finite group action against integral data.

    ``U(g) G_x U(g).T = sum_y Q(g)[y,x] G_y`` is the coupling covariance
    condition. Frequencies must commute with Q, so only degenerate modes may
    mix. Both actions must obey the supplied group multiplication, preserve
    orthogonality, and leave the electronic one-/two-body integrals invariant.
    This factory supports synthetic models as well as molecular adapters.
    """
    tolerance = _tolerance(tolerance)
    if not isinstance(model, ElectronPhononModel):
        raise TypeError("model must be an ElectronPhononModel")
    if not isinstance(group, PointGroupData):
        raise TypeError("group must be PointGroupData")
    size = group.order
    if (size < 1 or group.operations.shape != (size, 3, 3)
            or group.multiplication.shape != (size, size)
            or group.multiplication.dtype.kind not in "iu"
            or np.any(group.multiplication < 0) or np.any(group.multiplication >= size)):
        raise ValueError("group has invalid operation/multiplication arrays")
    owned_group = _group_data(group.name, group.operations, group.multiplication,
                              group.irrep_matrices)
    u = _real_array(orbital_actions, "orbital_actions")
    q = _real_array(photon_actions, "photon_actions")
    if u.shape != (size, model.norb, model.norb):
        raise ValueError("orbital_actions must have shape (G,norb,norb)")
    if q.shape != (size, model.nmode, model.nmode):
        raise ValueError("photon_actions must have shape (G,nmode,nmode)")
    for g in range(size):
        if not np.allclose(u[g] @ u, u[group.multiplication[g]],
                           atol=tolerance, rtol=0):
            raise ValueError("orbital actions violate group multiplication")
        if not np.allclose(q[g] @ q, q[group.multiplication[g]],
                           atol=tolerance, rtol=0):
            raise ValueError("photon actions violate group multiplication")
        error = _covariance_error(model, u[g], q[g], tolerance)
        if error is not None:
            raise ValueError(f"operation {g}: {error}")
    return JointSymmetry(owned_group, _owned(u, np.float64), _owned(q, np.float64))


def _restricted_group(parent, retained):
    """Use MolSym's subgroup identification/isomorphism, keeping lab axes."""
    if len(retained) == parent.order:
        return parent, np.asarray(retained, dtype=np.int64)
    _, generate, _, _ = _backend()
    from molsym.symtext.multiplication_table import identify_subgroup, is_subgroup, subgroup_by_name

    if not retained or not is_subgroup(retained, parent.multiplication):
        raise ValueError("cavity-invariant operations are not a closed subgroup; tighten tolerance")
    symels, _, _ = generate(parent.name)
    label = identify_subgroup(retained, symels)
    local_elements = [symels[g] for g in retained]
    inverse = np.full(parent.order, -1, dtype=np.int64)
    inverse[retained] = np.arange(len(retained))
    table = inverse[parent.multiplication[np.ix_(retained, retained)]]
    mapping = subgroup_by_name(local_elements, table, label)
    if mapping is None or len(mapping) != len(retained):
        raise ValueError("MolSym could not identify the complete cavity subgroup")
    ordered = np.array([retained[j] for _, j in sorted(mapping)], dtype=np.int64)
    _, _, matrices = generate(label)
    table = inverse[parent.multiplication[np.ix_(ordered, ordered)]]
    # Convert the restricted-table indices from retained order to canonical order.
    canonical = np.empty(len(retained), dtype=np.int64)
    canonical[inverse[ordered]] = np.arange(len(retained))
    group = _group_data(label, parent.operations[ordered], canonical[table], matrices)
    return group, ordered


def _mode_action(operation, scaled_polarization, frequencies, tolerance):
    """Return a block-orthogonal lift of O, or None if a block is not preserved."""
    count = len(frequencies)
    result = np.eye(count)
    remaining = set(range(count))
    while remaining:
        first = min(remaining)
        block = sorted(i for i in remaining
                       if abs(frequencies[i] - frequencies[first]) <= tolerance)
        remaining.difference_update(block)
        a = scaled_polarization[:, block]
        covariance = a @ a.T
        scale = np.max(np.abs(covariance))
        if scale == 0:
            continue
        if not np.allclose(operation @ covariance @ operation.T, covariance,
                           atol=tolerance * scale, rtol=0):
            return None
        inverse = np.linalg.pinv(a, rcond=tolerance)
        result[np.ix_(block, block)] = inverse @ operation @ a + np.eye(len(block)) - inverse @ a
    return result


def molecular_cavity_symmetry(model, *, subgroup=None, tolerance=1e-9):
    """Find the molecule's finite subgroup preserving its cavity Hamiltonian.

    A linear molecule requires an explicit finite parent ``subgroup``, such as
    D2h. Within each frequency block, columns of A are lambda_c*polarization.
    When O preserves A A.T, the photon action is A^+ O A + I - A^+ A; uncoupled
    photon directions therefore have the identity action. Further filtering
    checks the actual integrals, including dipole-origin terms. MolSym labels
    the retained subgroup and supplies its full irrep matrices. Nonclosed
    numerical selections raise rather than silently discard operations.
    """
    tolerance = _tolerance(tolerance)
    if not isinstance(model, MolecularPF):
        raise TypeError("model must be a MolecularPF")
    molecular = molecular_point_group(model.mol, subgroup=subgroup, tolerance=tolerance)
    if (model.ao_to_orth.shape != molecular.ao_to_orth.shape
            or not np.allclose(model.ao_to_orth, molecular.ao_to_orth,
                               atol=tolerance, rtol=0)):
        raise ValueError("the molecule/basis changed after the MolecularPF snapshot")
    a = (model.lambda_c[:, None] * model.polarization).T
    retained, photons = [], []
    for g, operation in enumerate(molecular.group.operations):
        q = _mode_action(operation, a, model.omega, tolerance)
        if q is not None and _covariance_error(
                model, molecular.orbital_actions[g], q, tolerance) is None:
            retained.append(g)
            photons.append(q)
    group, ordered = _restricted_group(molecular.group, retained)
    lookup = {g: q for g, q in zip(retained, photons)}
    q = np.stack([lookup[g] for g in ordered])
    return joint_symmetry(model, group, molecular.orbital_actions[ordered], q,
                          tolerance=tolerance)
