# MR_LF: Hubbard–Holstein Lang–Firsov methods

**Chenxi Wang & Zhihao Cui (advisor)**

A modular Python repository for exact diagonalization, coherent-state and
Lang–Firsov mean-field/MP2 calculations, fixed-frame LF-MR-NOCI, and
block-Hamiltonian LF-MRPT2. It contains the computational methods, explicit
examples, compact reference results and mathematical documentation.

The model has real hopping, one local phonon mode per site, fixed
`nelec=(N_alpha,N_beta)`, uncentered density coupling, and no phonon zero-point
energy. `alpha = g**2 / omega`. MR varies all electronic CI coefficients in
each supplied LF frame. Frame optimization is a separate HF calculation.

## Installation

Use Python 3.12 or newer on a platform supported by PySCF. The recorded
verification environment is Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1,
PySCF 2.14.0 and Matplotlib 3.11.1, on Linux. Other environments have not been
validated here.

From this directory:

```bash
python -m pip install ".[plot]"
```

For core calculations without plotting, use `python -m pip install .`.
`requirements.txt` records the direct dependency versions for calculations
and figures. The installed numerical stack supplies its BLAS/LAPACK backend;
last-digit differences and optimizer termination can depend on that backend.

## Minimal external call

```python
from mr_lf import HubbardHolstein, ExactED, CSHF, CSMP2, LFHF, LFMP2

model = HubbardHolstein.ring(4, nelec=(1, 0), alpha=0.0, omega=0.5)
ed = ExactED(model, nmax=2)
print(ed.kernel(), ed.result["residual"])  # E = -2

cs = CSHF(model)
cs.kernel()
print(CSMP2(cs).kernel())

lf = LFHF(model)
lf.kernel()
print(LFMP2(lf, max_total=4).kernel())
```

Every `kernel()` returns **total energy** and stores `e_tot`, `converged`,
and `result`. Post-HF methods also store `e_corr`. Results retain structured
coefficients and numerical diagnostics. `converged` refers to the numerical
calculation at the supplied cutoffs; it does not certify cutoff convergence.
Methods accepting orbitals use `[spin,site,MO]`, with occupied columns first.

For fixed-frame MR and PT2:

```python
import numpy as np
from mr_lf import LFMR, LFMRPT2

# Replace these zero-coupling demonstration seeds with chosen LF-HF frames.
lam = np.zeros((1, model.L, model.L), dtype=np.float64)
shift = np.zeros((1, model.L), dtype=np.float64)
mr = LFMR(model, lam, shift, character=1)  # momentum-zero projection
mr.kernel()
pt = LFMRPT2(mr, max_total=2, external_cut=1e-8)
print(pt.kernel(), pt.e_corr, pt.result["response_residual"])
```

`character=None` keeps unprojected frames; `+1` and `-1` select momentum
zero and pi, respectively. Pi requires an even, translation-invariant ring.
The examples show explicit multi-start selection and meaningful frame recipes.

## Methods and organization

| Method | Electron sectors | Reference / partition |
|---|---|---|
| `ExactED` | General fixed alpha/beta counts | Site determinants × truncated Fock basis |
| `CSHF` | Single and multiple electrons | Unrestricted determinant × coherent phonons |
| `CSMP2` | Single and multiple electrons | Canonical CS Fock; phonon singles and Hubbard doubles |
| `LFHF` | Single and multiple electrons | Full density-diagonal LF matrix, zero-shift gauge |
| `LFMP2` | Single and multiple electrons | Canonical single-reference LF partition |
| `LFMR` | Single and multiple electrons | Full site CI per fixed frame; optional translation projection |
| `LFMRPT2` | Single and multiple electrons | Complete retained reference/external Hamiltonian blocks |

The package groups model/basis/phonon tools, `exact`, `scf`, `mp`, `mr`,
`symmetry` and `observables`. `mr` separates analytic kernels, overlap
geometry, NOCI, response equations and precision repair. Computational
kernels do not read scan files or import examples. Low-level functions can
be imported directly; see [API documentation](docs/API.md).

## Molecular integrals and limited ED/LF-HF/MP2 support

`MolecularEP` obtains molecular electronic integrals directly from PySCF and
combines them with supplied linear harmonic-mode parameters. It uses a fixed
symmetric Löwdin electronic basis and allows different electronic orbital
and phonon mode counts. Constructing it does not run SCF:

```python
import numpy as np
from pyscf import gto
from mr_lf import ExactED, LFHF, LFMP2, MolecularEP

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
omega = np.array([0.5])  # Hartree
g_ao = np.zeros((1, mol.nao_nr(), mol.nao_nr()))  # replace with supplied coupling
model = MolecularEP(mol, omega=omega, coupling_ao=g_ao)
print(model.nelec, model.h1e.shape, model.eri.shape, model.coupling.shape)
# (1, 1) (2, 2) (2, 2, 2, 2) (1, 2, 2)
ed = ExactED(model, nmax=8, max_space=40, lindep=1e-22)
ed.kernel()
hf = LFHF(model, gtol=1e-8)
hf.kernel()
pt = LFMP2(hf, max_total=8)
pt.kernel()
```

`coupling_ao` contains AO matrix elements multiplying `b_x+b_x†`, in Hartree,
with any coordinate normalization already included. `model.ecore` stores
nuclear repulsion separately, with no phonon zero-point constant.
`ElectronPhononModel` accepts the corresponding orthonormal integral arrays
directly; `HubbardHolstein.to_general()` converts the existing lattice model.
Both new model types own copied arrays. Recreate a molecular model after
changing geometry or basis. Near-dependent AOs raise rather than being removed.

`ExactED`, `LFHF` and `LFMP2` accept both integral models. ED uses PySCF's
uncompressed electronic contractions and Davidson; LF-HF uses PySCF UHF/J/K
with analytic vacuum integrals and numerical joint orbital/LF optimization.
The first general LF-MP2 implementation is limited to **256 electronic
determinants**, includes all phonon-dressed electronic singles and doubles,
and requires positive canonical Fock excitation denominators. Phonon cutoffs
need separate convergence checks. CS, MR and MRPT2 remain lattice methods.
General gradients, Hessians, molecular symmetry restoration and MR reference
selection are subsequent work.
The [API](docs/API.md#general-integral-model-and-molecular-adapter) specifies
shapes and lifecycle, and the [theory](docs/theory.md#general-molecular-integral-interface-adapter-stage)
derives the basis and density maps.

Adapter acceptance is maintained outside the standalone release, alongside
the existing migration checks. From the parent research workspace, run:

```bash
python -m pytest -q tests/test_molecular_adapter.py tests/test_molecular_solvers.py tests/test_si_release.py
```

Adapter checks cover closed-shell H2 and open-shell
H3/STO-3G orthogonality, coupling expectations, same-state UHF energy and
PySCF FCI energy invariance, Hubbard mapping, input ownership and failure
cases. Solver checks independently construct spin-bit Hamiltonians and
large-oscillator displacement exponentials, compare zero-coupling energies
with PySCF FCI/UHF/UMP2, and verify symmetry partners and small perturbed starts.

For a small symmetry experiment, run from `si/`:

```bash
python -m examples.molecular_symmetry --output outputs/molecular_symmetry
```

This supplies an artificial odd mode `g*(n_0-n_1)*(b+b†)` to H2/STO-3G;
it is not a computed vibrational or cavity Hamiltonian. The CSV, JSON, frame
arrays and plots record initial-state dependence, charge/spin order parameters,
combined electronic/phonon parity, stationarity and both cutoff histories.
The [POLAR implementation](https://github.com/zhcui/polar_preview) provided
the general LF integral reference; it is not a runtime dependency.

## Reproduce the main figures

The compact results and frame parameters in `reference_data` allow immediate
redrawing. Run these commands from the repository directory:

```bash
python -m examples.plot_results
```

PDF and PNG files are written to `outputs/figures`. This command reads its
inputs and never runs a solver or rewrites reference results. It produces:

- Single- and multi-electron energy comparisons and errors relative to ED.
- Charge/spin density indicators, with ambiguous MR roots marked.
- MR frame-count/rank sensitivity and transferred broken-frame comparisons.
- MRPT2 phonon-cutoff histories and overlap-threshold sensitivity.

The single-electron grid is `alpha=0,0.2,...,3`, `omega=0.5`, `U=0`,
`nelec=(1,0)`. The multi-electron grid is `alpha=0,0.4,...,4`,
`omega=0.5,5`, `U=4`, `nelec=(2,2)`, always `L=4`, `t=-1`.
New multi-electron CS results are supplied separately in `multi_cs.csv`.
The [data guide](reference_data/README.md) explains frame recipes,
provenance, precision settings and convergence flags.

## Run calculations

Limit numerical threads for reproducible small-cluster timings:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m examples.single_electron_scan
python -m examples.multi_electron_scan
python -m examples.symmetry_breaking --case single --alpha 2.4
python -m examples.symmetry_breaking --case multi --alpha 2.4
python -m examples.mr_frame_convergence
python -m examples.mrpt2_cutoff --max-total 3 --overlap-sweep
python -m examples.cs_scan
```

The default scans use two alpha points and omit expensive ED and MRPT2.
Use `--full` for the complete grid and explicit `--with-ed` / `--with-mrpt2`
to include those methods. `--alphas` and `--omegas` select subsets. Examples:

```bash
python -m examples.single_electron_scan --alphas 0 0.4 --with-ed --with-mrpt2 --nmax 4 --max-total 2
python -m examples.multi_electron_scan --full --with-mrpt2 --max-total 5 --precision-mode reconditioned_longdouble
python -m examples.mrpt2_cutoff --case multi --omega 0.5 --alpha 0.8 --max-total 3 --precision-mode reconditioned_longdouble
```

To raise the MRPT2 cutoff at one archived point until two successive
increments `|E(M)-E(M-1)|` fall below `1e-6` (both external cuts), and to
redraw the converged scans:

```bash
python -m examples.mrpt2_m_convergence --case multi --omega 0.5 --alpha 2.4 --algorithm lean --output runs/multi/w0.5_a2.4.csv
python -m examples.plot_mrpt2_convergence --runs runs
```

`--algorithm lean` uses `mr_lf.mr.reduced`: exact translation-orbit
reduction of invariant seeds, the spin-exchange sector of the MR root, and
group-blocked overlap storage (`--defer-h` also contracts H_TT one group
pair at a time). It reproduces `LFMRPT2` energies to roundoff while using
several times less memory; reruns resume from the CSV.
The converged four-site scans (data, figures and provenance) are in
[results/mrpt2_phonon_cutoff_convergence](results/mrpt2_phonon_cutoff_convergence/README.md).

Scan calculations optimize their own references and record method failures;
they need not reproduce the last digits of archived local minima. To
recompute MR/PT2 using the exact archived frames, use:

```bash
python -m examples.reproduce_reference --case single --alpha 0.4
python -m examples.reproduce_reference --case multi --alpha 0
```

`--full` on that entry recomputes the complete archived MR/PT2 grid. Outputs
are separate from the immutable reference tables. The fixed-point MRPT2
example records every supplied phonon/overlap cutoff; it does not label an
arbitrary finite sequence as converged.

## Numerical conventions and practical limits

ED `nmax` limits each oscillator separately. MP2/MRPT2 `max_total` limits
the inclusive total occupation. Single-electron LF-MP2 includes all excited
modes; multi-electron LF-MP2 also exposes a mode-count restriction.

The bundled multi-electron M5 MRPT2 results passed their recorded numerical
checks; phonon-cutoff convergence remains unestablished. Open markers denote
finite-cutoff MRPT2 points. The single-electron table preserves its original
per-point cutoff and overlap-convergence flags. No incomplete high-M batch
is included. New multi-electron CS curves include all 22 grid points with
independent orbital stationarity checks.

MRPT2 uses a block-Hamiltonian partition and its corrected energy is not
variational. Unsafe reference gaps or external denominators raise errors.
The optional `reconditioned_longdouble` mode preserves retained ranks and
requires genuinely extended accumulation precision; it is verified on
Linux and raises if unsupported.

Dense MRPT2 matrices grow rapidly. Four sites, two frames and four electrons
at M5 produce a `9000 × 9000` candidate block; historical calculations used
about 3 GiB per worker. ED dimension is
`comb(L,N_alpha)*comb(L,N_beta)*(nmax+1)**L`, before Davidson workspace.
The complete multi-electron examples can take substantial time and memory;
the shipped tables support plotting without recalculation.

A centered ED calculation has a different finite-cutoff space. The
complete-space relation is
`E_paper = E_centered - g**2 * N_e**2 / (L * omega)`.
The scan's `--ed-convention centered` records this explicit shift but keeps
`ed_cutoff_converged=False` until independent cutoff evidence exists.

## Verification

The recorded release validation passed 211 checks: 182 existing numerical
reference checks and 29 SI migration/CS-extension checks. These cover
independent tiny Fock/bit-operator ED matrices, displacement exponentials,
old/new MR kernels and responses, PySCF Hubbard UHF/UMP2, the CS limit of
LF-MP2, density conservation and failure-state handling. Acceptance code is
maintained outside this standalone release.

Small single- and multi-electron scans including every method, both symmetry
examples, the frame-selection example, a phonon/overlap-cut sweep and a
fixed-frame single-electron reproduction were executed. The archived
single-electron alpha=0.4 result was reproduced to the printed precision.
The full historical ED/MRPT2 scans were not rerun; their evidence and
finite-cutoff flags are preserved. All 22 new multi-electron CS points were
computed. Reference hashes are recorded in `reference_data/manifest.json`.

## Theory, attribution and license

[Theory and algorithm derivations](docs/theory.md) give the index conventions,
all MR/MRPT2 matrix elements, displaced-number recurrences, projection signs,
reference subtraction, response equations and numerical design. The document
also derives the multi-electron CS extension and its limiting checks.

Single-reference LF methods are associated with [Cui, Mandal and Reichman
(2024)](https://doi.org/10.1021/acs.jctc.3c01166). Cite that work for those
methods and this repository for its implementation and MR extensions.
No manuscript DOI is assigned by this code release.

Code is distributed under [BSD-3-Clause](LICENSE).
