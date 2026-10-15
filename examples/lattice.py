"""Small Hubbard-Holstein calculation using public solver interfaces.

Run from the SI checkout: python -m examples.lattice.
Photon cutoffs are illustrative and require separate convergence studies.
"""

import json

import numpy as np
from pyscf import lib

from mr_lf import LFHF, LFMP2, LFMR, ExactED, HubbardHolstein


def main():
    with lib.with_omp_threads(1):
        model = HubbardHolstein.ring(4, nelec=(1, 0), alpha=.4, omega=.5)
        ed = ExactED(model, nmax=2, tol=1e-12, tol_residual=1e-10,
                     max_space=50, max_cycle=300, lindep=1e-22)
        ed.kernel()
        hf = LFHF(model)
        hf.kernel()
        mp2 = LFMP2(hf, max_total=2)
        mp2.kernel()
        lam = np.stack((np.zeros_like(hf.lam), hf.lam))
        mr = LFMR(model, lam, character=1)
        mr.kernel()
    result = {"energy_hartree": {"ED": ed.e_tot, "LFHF": hf.e_tot,
                                  "LFMP2": mp2.e_tot, "LFMR": mr.e_tot},
              "ed_nmax": 2, "mp2_max_total": 2,
              "ed_residual": float(ed.result["residual"]),
              "mr_rank": mr.result["rank"], "cutoff_convergence": "not checked"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
