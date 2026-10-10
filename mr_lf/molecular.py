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
    deleted. Existing Hubbard–Holstein solvers do not yet accept this model.
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
