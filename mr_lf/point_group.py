"""MolSym group data, PySCF AO actions and active electronic CI rotations.

The joint_symmetry and mr.point_group_kernels modules build on these utilities.
All operations are active: R(g) phi_p = sum_q phi_q U(g)[q,p].
Only real spherical, nonrelativistic PySCF AOs are adapted here.
"""

import re
from dataclasses import dataclass
from math import comb
from numbers import Real
from types import MappingProxyType

import numpy as np
from pyscf import gto, lo, symm
from pyscf.fci import addons
from pyscf.symm.Dmatrix import Dmatrix, get_euler_angles


def apply_electronic_symmetry(ci, norb, nelec, orbital_action):
    """Actively rotate fixed-electron-number CI in a common orthonormal basis.

    R a_p^dagger R^-1 = sum_q U[q,p] a_q^dagger, where U=orbital_action.
    PySCF's passive orbital-rotation interface therefore receives U.T.

    ci is a finite float64/complex128 ndarray with shape
    (comb(norb,N_alpha),comb(norb,N_beta)), in cistring address order.
    orbital_action is a finite real float64 orthogonal (norb,norb) ndarray.
    norb is an integer in [1,63]; nelec is an explicit integer pair tuple.
    Bools are not counts. Orthogonality uses atol=1e-12, rtol=0.

    Return an owned C-order array with the same shape/dtype as ci, without
    normalization. Read-only/noncontiguous inputs are supported and unchanged.
    Unsupported array types/dtypes raise TypeError; invalid scalars, shapes,
    nonfinite entries or nonorthogonality raise ValueError. No dense CI
    rotation matrix, LF transform, or photon group action is constructed.
    """
    if (isinstance(norb, (bool, np.bool_)) or not isinstance(norb, (int, np.integer))
            or not 1 <= norb < 64):
        raise ValueError("norb must be an integer in [1,63]")
    if (not isinstance(nelec, tuple) or len(nelec) != 2
            or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                   or not 0 <= n <= norb for n in nelec)):
        raise ValueError("nelec must be an integer (N_alpha,N_beta) tuple in [0,norb]")
    norb, nelec = int(norb), tuple(int(n) for n in nelec)
    if not isinstance(ci, np.ndarray) or ci.dtype not in (np.float64, np.complex128):
        raise TypeError("ci must be a float64 or complex128 NumPy array")
    if not isinstance(orbital_action, np.ndarray) or orbital_action.dtype != np.float64:
        raise TypeError("orbital_action must be a float64 NumPy array")
    if ci.shape != tuple(comb(norb, n) for n in nelec):
        raise ValueError("ci must have structured shape (comb(norb,Na),comb(norb,Nb))")
    if orbital_action.shape != (norb, norb):
        raise ValueError("orbital_action must have shape (norb,norb)")
    if not np.all(np.isfinite(ci)) or not np.all(np.isfinite(orbital_action)):
        raise ValueError("CI and orbital_action must contain only finite entries")
    with np.errstate(over="ignore", invalid="ignore"):
        gram = orbital_action.T @ orbital_action
    if not np.allclose(gram, np.eye(norb), atol=1e-12, rtol=0):
        raise ValueError("orbital_action must be orthogonal")
    # Retain the student's correct active-to-passive orientation.
    result = addons.transform_ci_for_orbital_rotation(ci, norb, nelec, orbital_action.T)
    return np.array(result, dtype=ci.dtype, order="C", copy=True)


def _backend():
    # MolSym's group module changes NumPy's global print options on import.
    # Keep this dependency's presentation settings out of the caller's session.
    with np.printoptions():
        try:
            import molsym
            from molsym.symtext.general_irrep_mats import pg_to_symels
            from molsym.symtext.multiplication_table import build_mult_table
            from molsym.symtext.point_group import PointGroup
        except ImportError as error:
            raise ImportError(
                'Point-group utilities require MolSym==1.2.0; install mr-lf[symmetry]'
            ) from error
    return molsym, pg_to_symels, build_mult_table, PointGroup


def _owned(values, dtype):
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PointGroupData:
    """Owned group data, in the backend's operation/irrep order.

    operations: float64 (G,3,3); multiplication: int64 (G,G), with
    operations[g] @ operations[h] = operations[multiplication[g,h]].
    irrep_matrices[label]: complex128 (G,d,d), including complex characters.
    Array values are read-only; no backend objects are exposed or mutated.
    """

    name: str
    operations: np.ndarray
    multiplication: np.ndarray
    irrep_matrices: MappingProxyType

    @property
    def order(self):
        return len(self.operations)


def _group_data(name, operations, multiplication, matrices):
    operations = _owned(operations, np.float64)
    multiplication = _owned(multiplication, np.int64)
    irreps = {label: _owned(matrix, np.complex128) for label, matrix in matrices.items()}
    group = PointGroupData(name, operations, multiplication, MappingProxyType(irreps))
    if not np.all(np.isfinite(operations)):
        raise ValueError("backend returned nonfinite operations")
    if not np.allclose(operations[:, None] @ operations[None, :],
                       operations[multiplication], atol=1e-10, rtol=0):
        raise ValueError("backend spatial operations violate group multiplication")
    if not np.allclose(operations.transpose(0, 2, 1) @ operations,
                       np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("backend spatial operations are not orthogonal")
    characters, dimensions = [], []
    for label, matrix in irreps.items():
        d = matrix.shape[1]
        if matrix.shape != (group.order, d, d) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"invalid backend irrep {label}")
        if not np.allclose(matrix[:, None] @ matrix[None, :],
                           matrix[multiplication], atol=1e-10, rtol=0):
            raise ValueError(f"backend irrep {label} violates group multiplication")
        if not np.allclose(matrix.conj().transpose(0, 2, 1) @ matrix,
                           np.eye(d), atol=1e-10, rtol=0):
            raise ValueError(f"backend irrep {label} is not unitary")
        characters.append(np.trace(matrix, axis1=1, axis2=2))
        dimensions.append(d)
    chars = np.asarray(characters)
    if (sum(d*d for d in dimensions) != group.order
            or not np.allclose(chars.conj() @ chars.T / group.order,
                               np.eye(len(chars)), atol=1e-10, rtol=0)):
        raise ValueError("backend irreps fail completeness/character orthogonality")
    return group


def point_group_data(name):
    """Get validated finite point-group data from MolSym, without a molecule.

    Supports the finite molecular point-group families supplied by MolSym.
    Continuous groups (e.g. D0h) require a finite subgroup instead. This
    function generates no electronic determinants, Hamiltonians or projectors.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a Schoenflies string")
    if name in ("C0v", "D0h", "Coov", "Dooh", "K", "Kh"):
        raise ValueError("continuous point groups require an explicit finite subgroup")
    if re.fullmatch(r"(?:C(?:[1-9]\d*[vh]?|s|i)|D[1-9]\d*[hd]?|S[1-9]\d*|T[dh]?|Oh?|Ih?)",
                    name) is None:
        raise ValueError("name must be a canonical finite Schoenflies symbol, e.g. C3v")
    _, generate, multiplication, point_group = _backend()
    try:
        parsed = point_group.from_string(name)
        if parsed.is_linear:
            raise ValueError("continuous point groups require an explicit finite subgroup")
        symels, _, matrices = generate(parsed.str)
    except Exception as error:
        raise ValueError(f"unsupported point group {name!r}") from error
    return _group_data(parsed.str, [s.rrep for s in symels],
                       multiplication(symels), matrices)


@dataclass(frozen=True)
class MolecularPointGroup:
    """Snapshot of a molecule's finite group and active basis actions.

    group.operations act in the original lab frame, around origin (Bohr).
    atom_map[g,i] is the destination of atom i. ao_actions[g,q,p] are
    nonorthogonal-AO coefficient actions and preserve the AO metric S.
    ao_to_orth is symmetric Lowdin X; orbital_actions = X^-1 U_AO X
    are orthogonal in the fixed orthonormal basis. All arrays are float64
    except atom_map (int64). This snapshot does not follow Mole changes.
    """

    group: PointGroupData
    origin: np.ndarray
    atom_map: np.ndarray
    overlap_ao: np.ndarray
    ao_to_orth: np.ndarray
    ao_actions: np.ndarray
    orbital_actions: np.ndarray


def _atom_mapping(mol, operations, origin, tolerance):
    coords = mol.atom_coords() - origin
    symbols = np.array([mol.atom_pure_symbol(i) for i in range(mol.natm)])
    mapping = np.empty((len(operations), mol.natm), dtype=np.int64)
    for g, operation in enumerate(operations):
        moved = coords @ operation.T
        for i, position in enumerate(moved):
            matches = np.flatnonzero((symbols == symbols[i])
                                    & (np.linalg.norm(coords - position, axis=1) < tolerance))
            if len(matches) != 1:
                raise ValueError("group does not uniquely map the original nuclear geometry")
            mapping[g, i] = matches[0]
        if len(set(mapping[g])) != mol.natm:
            raise ValueError("atom mapping must be a permutation")
    return mapping


def _ao_actions(mol, operations, atom_map):
    shells = [[b for b in range(mol.nbas) if mol.bas_atom(b) == a]
              for a in range(mol.natm)]
    ao_loc = mol.ao_loc_nr()
    actions = np.zeros((len(operations), mol.nao_nr(), mol.nao_nr()))
    for g, operation in enumerate(operations):
        inversion = np.linalg.det(operation) < 0
        proper = -operation if inversion else operation
        angles = get_euler_angles(np.eye(3), proper.T)
        angular = {}
        for source, destination in enumerate(atom_map[g]):
            if len(shells[source]) != len(shells[destination]):
                raise ValueError("symmetry-equivalent atoms have different AO bases")
            for src, dst in zip(shells[source], shells[destination]):
                l = mol.bas_angular(src)
                nctr = mol.bas_nctr(src)
                if (l != mol.bas_angular(dst) or nctr != mol.bas_nctr(dst)
                        or not np.array_equal(mol.bas_exp(src), mol.bas_exp(dst))
                        or not np.array_equal(mol.bas_ctr_coeff(src), mol.bas_ctr_coeff(dst))):
                    raise ValueError("symmetry-equivalent atoms have different AO bases")
                if l not in angular:
                    angular[l] = Dmatrix(l, *angles, reorder_p=True)
                    if inversion:
                        angular[l] *= (-1)**l
                width = 2*l + 1
                for contraction in range(nctr):
                    p = ao_loc[src] + contraction*width
                    q = ao_loc[dst] + contraction*width
                    actions[g, q:q+width, p:p+width] = angular[l]
    return actions


def molecular_point_group(mol, *, subgroup=None, tolerance=1e-8):
    """Adapt a built PySCF Mole to MolSym data and active AO/Lowdin actions.

    No SCF is run; the input Mole, its coordinates, and its basis are unchanged.
    Geometry tolerance is in Bohr. Approximate geometries are never repaired.
    Spherical AOs are required. A finite subgroup must be specified for a
    linear molecule (e.g. subgroup='D2h' for H2). For nonlinear molecules,
    subgroup optionally requests a MolSym-supported subgroup embedding.

    This stage has no photon-mode representation, coupling invariance filter,
    LF parameter transformation, or molecular LFMR driver integration.
    """
    if not isinstance(mol, gto.Mole):
        raise TypeError("mol must be a pyscf.gto.Mole")
    if not mol._built or not mol.natm or not mol.nao_nr():
        raise ValueError("mol must be built with atoms and a nonempty AO basis")
    if mol.cart:
        raise NotImplementedError("the first symmetry stage requires spherical AOs")
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be a finite positive real scalar in Bohr")
    if subgroup is not None and not isinstance(subgroup, str):
        raise TypeError("subgroup must be a finite Schoenflies string or None")
    requested = point_group_data(subgroup) if subgroup is not None else None
    backend, _, _, _ = _backend()
    geometry = backend.Molecule(
        [mol.atom_pure_symbol(i) for i in range(mol.natm)],
        mol.atom_coords().copy(), mol.atom_mass_list().copy())
    geometry.tol = float(tolerance)
    origin = geometry.find_com()
    geometry.translate(origin)
    detected, _ = backend.find_point_group(geometry)
    if detected in ("D0h", "C0v"):
        if subgroup is None:
            raise ValueError("linear molecules require an explicit finite subgroup, e.g. D2h")
        # Avoid MolSym 1.2.0's continuous-Symtext path (no finite multiplication
        # table). PySCF supplies the molecular axis; MolSym supplies the finite group.
        _, origin, axes = symm.geom.detect_symm(mol._atom, mol._basis)
        back = axes.T
        standard = requested
    else:
        context = backend.Symtext.from_molecule(geometry)
        back = context.reverse_rotate
        if subgroup is not None and subgroup != context.pg.str:
            try:
                child = context.subgroup_symtext(subgroup)
            except Exception as error:
                raise ValueError(f"{subgroup} is not an available subgroup of {context.pg.str}") from error
            back = back @ child.reverse_rotate
            context = child
        standard = requested if requested is not None else point_group_data(context.pg.str)
    operations = back @ standard.operations @ back.T
    group = _group_data(standard.name, operations, standard.multiplication,
                        standard.irrep_matrices)
    atom_map = _atom_mapping(mol, operations, origin, float(tolerance))
    actions = _ao_actions(mol, operations, atom_map)
    overlap = mol.intor_symmetric("int1e_ovlp")
    values = np.linalg.eigvalsh(overlap)
    if values[-1] <= 0 or values[0] <= 1e-10*values[-1]:
        raise ValueError("AO overlap must be positive and well conditioned; no AOs are deleted")
    x = lo.orth.lowdin(overlap)
    orth = (x.T @ overlap) @ actions @ x
    if not np.allclose(actions.transpose(0, 2, 1) @ overlap @ actions,
                       overlap, atol=1e-10, rtol=0):
        raise ValueError("AO actions do not preserve the molecular overlap metric")
    if not np.allclose(orth.transpose(0, 2, 1) @ orth,
                       np.eye(mol.nao_nr()), atol=1e-10, rtol=0):
        raise ValueError("Lowdin orbital actions violate orthogonality")
    # Do not allocate a (G,G,norb,norb) multiplication check tensor.
    for g in range(group.order):
        if not np.allclose(orth[g] @ orth, orth[group.multiplication[g]],
                           atol=1e-10, rtol=0):
            raise ValueError("Lowdin orbital actions violate group multiplication")
    return MolecularPointGroup(
        group, _owned(origin, np.float64), _owned(atom_map, np.int64),
        _owned(overlap, np.float64), _owned(x, np.float64),
        _owned(actions, np.float64), _owned(orth, np.float64))
