"""PySCF Mole to fixed symmetric-Löwdin electronic–phonon integral data."""

import numpy as np
from pyscf import ao2mo, gto, lo, scf

from .model import ElectronPhononModel, _real_array, _symmetric_array, _validate_phonons


class MolecularEP(ElectronPhononModel):
    r"""Snapshot a built, nonrelativistic PySCF Mole without running SCF.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Built molecular object supplying the full electronic basis, electron
        counts and nuclear repulsion energy. The adapter keeps this object
        by reference but never builds it or changes it. Recreate the model
        after changing the molecule; integral arrays do not update themselves.
    omega : array_like, shape (nmode,)
        Positive frequencies in Hartree; may be empty for a purely electronic
        model. No boson zero-point energy is included.
    coupling_ao : array_like, shape (nmode,nao,nao)
        Real symmetric AO matrix elements multiplying b_x+b_x^dagger, in
        Hartree, including any desired coordinate normalization. These are
        not raw coordinate derivatives. No mass/frequency factor is added.

    Attributes
    ----------
    overlap_ao : float64 ndarray, shape (nao,nao)
        Copied AO metric S at construction.
    ao_to_orth : float64 ndarray, shape (nao,norb)
        X=S^(-1/2), computed by PySCF. X.T @ S @ X = I, with all AOs kept.
        C_AO = X @ C_orth and D_AO = X @ D_orth @ X.T.
    orthogonalization : str
        "lowdin" (symmetric Löwdin, not meta-Löwdin).

    Other attributes follow ElectronPhononModel. ecore is mol.energy_nuc(),
    to be included once in total energies. The overlap eigenvalue ratio
    must be greater than 1e-10; near-dependent AOs raise rather than being
    deleted. ExactED, LFHF, LFMP2 and fixed-frame LFMR accept this model.
    """

    def __init__(self, mol, *, omega, coupling_ao):
        if not isinstance(mol, gto.Mole):
            raise TypeError("mol must be a pyscf.gto.Mole")
        if not mol._built:
            raise ValueError("mol must already be built; call mol.build() before adapting it")
        norb = mol.nao_nr()
        if norb == 0:
            raise ValueError("mol must have a nonempty electronic basis")
        omega, coupling_ao = _validate_phonons(
            omega, coupling_ao, norb, coupling_name="coupling_ao"
        )
        overlap = _real_array(mol.intor_symmetric("int1e_ovlp"), "overlap_ao")
        if overlap.shape != (norb, norb):
            raise ValueError("overlap_ao must have shape (nao,nao)")
        overlap = _symmetric_array(overlap, "overlap_ao")
        eigenvalues = np.linalg.eigvalsh(overlap)
        if eigenvalues[-1] <= 0 or eigenvalues[0] <= 1e-10 * eigenvalues[-1]:
            raise ValueError(
                "AO overlap is nonpositive or near linearly dependent: "
                "minimum/maximum eigenvalue ratio must exceed 1e-10; no AOs are removed"
            )

        x = lo.orth.lowdin(overlap)
        h1e = x.T @ scf.hf.get_hcore(mol) @ x
        eri = ao2mo.kernel(mol, x, compact=False).reshape((norb,) * 4)
        coupling = np.einsum("up,xuv,vq->xpq", x, coupling_ao, x, optimize=True)
        super().__init__(h1e, eri, omega, coupling, mol.nelec, ecore=mol.energy_nuc())
        self.mol = mol
        self.overlap_ao = overlap
        self.ao_to_orth = np.array(x, dtype=np.float64, order="C", copy=True)
        self.orthogonalization = "lowdin"


class MolecularPF(MolecularEP):
    r"""Dipole-gauge molecular Pauli–Fierz integrals with Cui's DSE convention.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Built, all-electron molecule with a nonempty real AO basis and a
        positive fixed electron number. No SCF is run. ECP and pseudopotential
        molecules are rejected because their nuclear-charge convention is not
        supported here. The supplied molecule, including its integral origin,
        is unchanged.
    omega : array_like, shape (nmode,)
        Positive photon frequencies, in Hartree. An empty array is allowed.
        The Hamiltonian contains omega*b^dagger*b without a zero-point term.
    polarization : array_like, shape (nmode,3)
        Real Cartesian unit vectors. Squared norms must equal one within
        1e-12; vectors are not normalized silently.
    lambda_c : array_like, shape (nmode,)
        Finite nonnegative coupling strengths in atomic units.
    origin : array_like, shape (3,), optional
        Common electronic/nuclear dipole origin, in Bohr, irrespective of
        mol.unit. Defaults to the nuclear-charge center.

    Notes
    -----
    Let r_k be the position matrix about origin, in the same symmetric-Lowdin
    basis as all electronic integrals, and d_nuc=sum_A Z_A*(R_A-origin).
    For the fixed N_e sector define

        mu_k = r_k - d_nuc[k]/N_e * I,
        mu_x = sum_k polarization[x,k]*mu_k.

    Then sum_pq,sigma mu_k[p,q] a_p,sigma^dagger a_q,sigma is precisely the
    electronic position operator minus the nuclear dipole constant. This
    follows Cui's sign convention for mu; it is the negative of the usual
    physical molecular charge dipole. The division by N_e distributes the
    nuclear constant once, not once per electron. Consequently this model
    must not be reused in a different electron-number sector.

    The coupling and dipole self-energy (DSE) are

        g_x = -sqrt(omega_x/2)*lambda_c[x]*mu_x,
        delta_h = 1/2 sum_x lambda_c[x]^2*(mu_x @ mu_x),
        delta_eri[p,q,r,s] = sum_x lambda_c[x]^2*mu_x[p,q]*mu_x[r,s].

    Thus the electronic DSE is 1/2 sum_x lambda_c[x]^2*mu_hat_x^2 in
    the finite electronic space. The product of dipole matrices is formed
    after orthogonalization; no quadrupole-integral correction is included.
    Nuclear dipole terms are already encoded in mu, so ecore remains the
    nuclear repulsion alone. These choices implement Cui Eqs. (5), (7)-(9).

    Attributes
    ----------
    origin, nuclear_dipole : float64 ndarray, shape (3,)
        Dipole origin and nuclear position dipole, in atomic units.
    polarization, lambda_c : float64 ndarray
        Owned copies of the supplied mode parameters.
    dipole_orth : float64 ndarray, shape (3,norb,norb)
        Effective mu_k including the distributed nuclear constant.
    projected_dipole : float64 ndarray, shape (nmode,norb,norb)
        The polarization contractions mu_x.
    h1e_dse : float64 ndarray, shape (norb,norb)
        DSE correction already included in h1e. The four-index DSE correction
        is not stored separately; eri already includes it.
    dse_convention : str
        "projected_dipole_product". All other attributes follow MolecularEP.
    """

    def __init__(self, mol, *, omega, polarization, lambda_c, origin=None):
        # Validate scalar/shape parameters before requesting any AO integrals.
        if not isinstance(mol, gto.Mole):
            raise TypeError("mol must be a pyscf.gto.Mole")
        if not mol._built:
            raise ValueError("mol must already be built; call mol.build() before adapting it")
        norb = mol.nao_nr()
        if norb == 0:
            raise ValueError("mol must have a nonempty electronic basis")
        nelectron = sum(mol.nelec)
        if nelectron <= 0:
            raise ValueError("MolecularPF requires a positive fixed electron number")
        if mol.has_ecp() or mol.pseudo or getattr(mol, "_pseudo", None):
            raise ValueError("MolecularPF currently requires an all-electron molecule without ECP/pseudo")

        omega = _real_array(omega, "omega")
        if omega.ndim != 1 or np.any(omega <= 0):
            raise ValueError("omega must be a one-dimensional array of positive frequencies")
        polarization = _real_array(polarization, "polarization")
        if polarization.shape != (omega.size, 3):
            raise ValueError("polarization must have shape (nmode,3)")
        # Squared norms avoid an unnecessary square root and cannot hide a
        # nonfinite/overflowing vector behind normalization.
        with np.errstate(over="ignore", invalid="ignore"):
            norm_squared = np.sum(polarization * polarization, axis=1)
        if not np.allclose(norm_squared, 1.0, atol=1e-12, rtol=0):
            raise ValueError("each polarization must be a unit vector (squared-norm tolerance 1e-12)")
        lambda_c = _real_array(lambda_c, "lambda_c")
        if lambda_c.shape != (omega.size,) or np.any(lambda_c < 0):
            raise ValueError("lambda_c must have shape (nmode,) and be nonnegative")

        charges = _real_array(mol.atom_charges(), "nuclear_charges")
        coordinates = _real_array(mol.atom_coords(unit="Bohr"), "nuclear_coordinates")
        if origin is None:
            if np.sum(charges) <= 0:
                raise ValueError("the default dipole origin requires positive total nuclear charge")
            with np.errstate(over="raise", invalid="raise"):
                try:
                    origin = np.einsum("a,ak->k", charges, coordinates) / np.sum(charges)
                except FloatingPointError as error:
                    raise ValueError("nuclear-charge center must be finite") from error
        origin = _real_array(origin, "origin")
        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,) in Bohr")

        # Dipole integration is isolated on a copy because PySCF stores the
        # common origin in the Mole environment even inside its context API.
        dipole_mol = mol.copy()
        with dipole_mol.with_common_orig(origin):
            position_ao = _real_array(
                dipole_mol.intor_symmetric("int1e_r", comp=3), "position_ao"
            )
        position_ao = _symmetric_array(position_ao, "position_ao")
        overlap = _real_array(mol.intor_symmetric("int1e_ovlp"), "overlap_ao")
        with np.errstate(over="raise", invalid="raise"):
            try:
                nuclear_dipole = np.einsum("a,ak->k", charges, coordinates - origin)
                dipole_ao = position_ao - (
                    nuclear_dipole[:, None, None] / nelectron * overlap
                )
                projected_ao = np.einsum("xk,kpq->xpq", polarization, dipole_ao)
                # sqrt(omega)*sqrt(1/2) also handles a positive subnormal omega
                # without rounding omega/2 prematurely to zero.
                coupling_ao = -(lambda_c[:, None, None] * projected_ao) * (
                    np.sqrt(omega)[:, None, None] * np.sqrt(0.5)
                )
            except FloatingPointError as error:
                raise ValueError("PF dipole/coupling arithmetic must remain finite") from error
        nuclear_dipole = _real_array(nuclear_dipole, "nuclear_dipole")
        dipole_ao = _real_array(dipole_ao, "dipole_ao")
        coupling_ao = _real_array(coupling_ao, "coupling_ao")

        # Reuse the electronic integral, overlap validation, and Lowdin path.
        super().__init__(mol, omega=omega, coupling_ao=coupling_ao)
        with np.errstate(over="raise", invalid="raise"):
            try:
                x = self.ao_to_orth
                dipole_orth = np.einsum("up,kuv,vq->kpq", x, dipole_ao, x, optimize=True)
                # Symmetry is exact algebraically; explicitly remove only the
                # multiplication-order roundoff before taking operator products.
                dipole_orth = dipole_orth * 0.5 + dipole_orth.transpose(0, 2, 1) * 0.5
                projected_dipole = np.einsum("xk,kpq->xpq", polarization, dipole_orth)
                scaled_dipole = lambda_c[:, None, None] * projected_dipole
                h1e_dse = 0.5 * np.einsum("xpr,xrq->pq", scaled_dipole, scaled_dipole)
                h1e_dse = h1e_dse * 0.5 + h1e_dse.T * 0.5
                h1e = self.h1e + h1e_dse
                eri = self.eri + np.einsum("xpq,xrs->pqrs", scaled_dipole, scaled_dipole)
            except FloatingPointError as error:
                raise ValueError("PF dipole self-energy arithmetic must remain finite") from error
        self.h1e = _real_array(h1e, "h1e including DSE")
        self.eri = _real_array(eri, "eri including DSE")
        self.origin = origin
        self.nuclear_dipole = nuclear_dipole
        self.polarization = polarization
        self.lambda_c = lambda_c
        self.dipole_orth = _real_array(dipole_orth, "dipole_orth")
        self.projected_dipole = _real_array(projected_dipole, "projected_dipole")
        self.h1e_dse = _real_array(h1e_dse, "h1e_dse")
        self.dse_convention = "projected_dipole_product"
