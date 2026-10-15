"""H2 cavity LFHF, projected LFMR and fixed-pool selection via public APIs.

Run from the SI checkout: python -m examples.molecular.
This is a usage example at STO-3G, not a reference-selection benchmark.
"""

import json

import numpy as np
from pyscf import gto, lib

from mr_lf import LFHF, LFMR, MolecularPF, molecular_cavity_symmetry, select_lf_references


def main():
    with lib.with_omp_threads(1):
        mol = gto.M(atom="H 0 0 -.37; H 0 0 .37", basis="sto-3g",
                    unit="Angstrom", charge=0, spin=0, verbose=0)
        model = MolecularPF(mol, omega=[.466751], polarization=[[0., 0., 1.]],
                            lambda_c=[.05])
        hf = LFHF(model, gtol=1e-8, max_cycle=1000)
        orbitals = np.stack([np.linalg.eigh(model.h1e)[1]] * 2)
        lam0 = np.diagonal(model.coupling, axis1=1, axis2=2) / model.omega[:, None]
        hf.kernel(mo_coeff0=orbitals, lam0=lam0)
        group = molecular_cavity_symmetry(model, subgroup="D2h")
        lam = np.stack((np.zeros_like(hf.lam), hf.lam))
        mr = LFMR(model, lam, symmetry=group, irrep="A_1g")
        mr.kernel()
        diagnostics = mr.symmetry_diagnostics()
        selected = select_lf_references(model, lam, symmetry=group, irrep="A_1g")
    result = {"energy_hartree": {"LFHF": hf.e_tot, "LFMR": mr.e_tot},
              "group": group.group.name, "irrep": "A_1g", "mr_rank": mr.result["rank"],
              "mr_residual": mr.result["residual_projected"],
              "purity_error": diagnostics["representation_purity_error"],
              "selected_frames": selected["chosen_indices"]}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
