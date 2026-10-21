# MR_LF: Lang–Firsov methods and molecular symmetry restoration

**Chenxi Wang & Zhihao Cui (advisor)**

A standalone Python library for exact diagonalization, coherent-state and
Lang–Firsov HF/MP2, fixed-frame LFMR and lattice block-Hamiltonian MRPT2.
Molecular cavities use PySCF integrals and optional MolSym finite-group
projection. This release contains the library, English theory/API guides,
small usage examples and compact reference data for the published figures.
Research scans, independent oracles and automated tests live in the separate
[research repository](https://github.com/SunsetStand/my_lf_mrci).

## Install

Use Python 3.12 or newer on a platform supported by PySCF. The tested
numerical environment is Linux, Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1
and PySCF 2.14.0. From this checkout:

```bash
python -m pip install .
# Add molecular point groups and figure plotting when needed:
python -m pip install ".[symmetry,plot]"
```

MolSym 1.2.0 supplies group data and is imported lazily. The core library
has no pytest, research-repository or benchmark dependency. Exact versions
used for calculations are listed in `requirements.txt`. Keep numerical
threads small for these examples:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## First lattice calculation

```python
import numpy as np
from mr_lf import ExactED, HubbardHolstein, LFHF, LFMP2, LFMR

model = HubbardHolstein.ring(4, nelec=(1, 0), alpha=0., omega=0.5)
ed = ExactED(model, nmax=2)
print(ed.kernel())  # -2 Hartree
hf = LFHF(model)
hf.kernel()
print(LFMP2(hf, max_total=2).kernel())
lam = np.stack((np.zeros_like(hf.lam), hf.lam))
mr = LFMR(model, lam, character=1)
print(mr.kernel(), mr.result["rank"])
```

Each `kernel()` returns total energy and stores `e_tot`, `converged` and
`result`. Post-HF methods also store `e_corr`. A candidate LF frame spans
independent electronic CI coefficients; frame optimization is a separate
HF calculation. Lattice `character=+1/-1` selects momentum zero/pi; omit
it for unprojected frames. Pi requires an even, translation-invariant ring.

## First molecular cavity calculation

```python
import numpy as np
from pyscf import gto
from mr_lf import MolecularPF, LFMR, molecular_cavity_symmetry

mol = gto.M(atom="H 0 0 -.37; H 0 0 .37", unit="Angstrom",
            basis="sto-3g", charge=0, spin=0, verbose=0)
model = MolecularPF(mol, omega=[0.466751], polarization=[[0., 0., 1.]],
                    lambda_c=[0.05])
group = molecular_cavity_symmetry(model, subgroup="D2h")
print(list(group.group.irrep_matrices))  # MolSym's exact labels
lam = np.zeros((1, model.nmode, model.norb), dtype=np.float64)
mr = LFMR(model, lam, symmetry=group, irrep="A_1g")
print(mr.kernel(), mr.symmetry_diagnostics()["representation_purity_error"])
```

`MolecularPF` uses symmetric Lowdin orbitals, the electron-minus-nuclear
position dipole and Cui's finite-basis dipole-product DSE. The default
origin is the nuclear-charge center. Nuclear repulsion is included once;
photon zero-point energy is omitted. Constructing the adapter runs no SCF.
Rebuild it after changing geometry, basis or electron number.

`MolecularEP` accepts supplied AO coupling matrices, and
`ElectronPhononModel` accepts orthonormal integral arrays directly. General
integral models support `ExactED`, `LFHF`, `LFMP2` and fixed-frame `LFMR`.
Joint point-group projection allows degenerate photon modes to mix and
handles non-Abelian partners and complex irreps. An optional common
selected-determinant list must be closed under the electronic group.
`select_lf_references` compares unprojected single-frame ranking, projected
ranking and actual MR-gain greedy selection on a fixed supplied pool.

The [molecular guide](docs/molecular_symmetry.md) derives the conventions,
projection formulas, array shapes, reference selection and reported study.
The [API](docs/API.md) gives exact signatures and limits. Molecular MRPT2,
VAP, Hessians, spin projection and projected molecular RDMs are not implemented.

## Examples and figures

Run these short usage examples from this source checkout after installation:

```bash
python -m examples.lattice
python -m examples.molecular
python -m examples.plot_results --output outputs/figures
```

The first two call public solvers. The plotting example reads immutable
`reference_data/` tables and writes PNG/PDF figures without running scans.
Examples and reference data are supplied in the source distribution;
Python wheels contain the library. See the [data guide](reference_data/README.md)
for convergence flags, frame recipes and historical provenance.

## Numerical conventions and scope

Lattice density coupling is uncentered, `alpha=g**2/omega`, with no phonon
zero-point constant. Integral arrays are real, with chemists' full ERIs;
projected CI coefficients may be complex. `nelec=(N_alpha,N_beta)` is explicit.
Electronic and phonon tensor axes remain structured until a solver boundary.

ED `nmax` is a per-mode cutoff. MP2/MRPT2 `max_total` limits the inclusive
total number. A numerically converged solve does not establish cutoff
convergence. Total-number photon truncation is needed for an independent
oracle preserving degenerate-mode rotations; that research helper lives
outside the release. Infinite coherent MR and finite-cutoff ED have
different variational spaces.

General LFMP2 and molecular LFMR support at most 256 electronic determinants;
LFMR supports at most 1024 projected pool coordinates. Dense lattice MRPT2
can require gigabytes. Its corrected energy is not variational, and unsafe
gaps/denominators raise. `reconditioned_longdouble` requires extended local
precision. The merged `mr_lf.mr.reduced` provides translation/spin-exchange
reduction and grouped/deferred-H storage for small-memory lattice MRPT2;
see [its API](docs/API.md#symmetry-reduced-high-cutoff-mrpt2). It supports
float64 and requires explicit cutoff and residual checks.

Historical multi-electron M5 MRPT2 data have numerical acceptance but no
established photon convergence; their flags remain unchanged. No larger
basis or complete candidate-space convergence is implied by this release.

## Converged Hubbard–Holstein MRPT2 results

The [four-site phonon-cutoff study](results/mrpt2_phonon_cutoff_convergence/README.md)
contains the original HPC histories, summary tables and final PNG/PDF figures.
All 16 single-electron and 22 four-electron points meet two successive
`|E(M)-E(M-1)| < 1e-6` at external overlap cut `1e-8`, with `3 <= M* <= 14`.
Failures at the tighter `1e-10` cut remain recorded separately.
These histories provide the later cutoff-converged results; the earlier
finite-cutoff tables in `reference_data/` retain their original values and flags.

To redraw from the sibling research checkout (with this repository at `si/`):

```bash
python -m benchmarks.si.plot_mrpt2_convergence --runs si/results/mrpt2_phonon_cutoff_convergence/history --output-dir outputs/mrpt2_m_converged/figures
```

## Research workflows and verification

Automated regression tests, cutoff scans, multistart searches, independent
bit/Fock oracles, resource probes and convergence-driver plots are maintained
in [the research workspace](https://github.com/SunsetStand/my_lf_mrci).
Its [benchmarks/si guide](https://github.com/SunsetStand/my_lf_mrci/tree/main/benchmarks/si)
contains the migrated workflows, including Claude's MRPT2 M-convergence
and plot drivers. For example, from that checkout:

```bash
python -m pytest -q
python -m benchmarks.si.molecular_lfmr --suite --output outputs/lfmr_suite
python -m benchmarks.si.mrpt2_m_convergence --help
```

These tools are not imported by `mr_lf`. The first H2/H3+ study and its
recorded acceptance are described in the English molecular guide. The
research repository keeps raw results and execution-source snapshots;
file moves do not rewrite historical energy or provenance records.

## Theory, attribution and license

[Theory](docs/theory.md) derives Hamiltonians, LF kernels, projection,
nonorthogonal geometry, MRPT2 response and precision handling. [API](docs/API.md)
contains runnable contracts. Single-reference LF methods follow
[Cui, Mandal and Reichman (2024)](https://doi.org/10.1021/acs.jctc.3c01166).
Cite that work for those methods and this repository for the implementation
and MR extensions. No manuscript DOI is assigned by this release.

Code is distributed under [BSD-3-Clause](LICENSE).
