# Public API

**Chenxi Wang & Zhihao Cui (advisor)**

## Common contract

Import calculation objects from `mr_lf`. `kernel()` returns a Python float
**total energy**. The object stores `e_tot`, post-reference `e_corr`,
`converged`, and a method-specific `result` dictionary containing numerical
arrays and diagnostics. `kernel()` clears previous results before work;
failed calls do not leave a previous energy marked successful.

Physical object inputs are real. The model owns a copied, symmetric
`float64` hopping matrix; supplied orbital/frame arrays must be finite
`float64`. Integer occupations are `int64`. Arrays use C order.
Low-level functions retain their documented dtype/shape contracts.
Results are ordinary NumPy/scalar dictionaries and are not automatically
JSON-serializable.

## Model

`HubbardHolstein(tmat, U, g, omega, nelec)` represents uncentered coupling
with one phonon mode per site and no zero-point constant.
`tmat.shape=(L,L)`, `0<L<64`, `omega>0`,
`nelec=(N_alpha,N_beta)`, `0<=N_sigma<=L`.
Properties are `L` and `alpha=g**2/omega`.

`HubbardHolstein.ring(L, *, nelec=(1,0), t=-1, U=0, omega=0.5, alpha=None, g=None)`
constructs a periodic ring with `L>=3`. Specify either alpha or g;
if both are absent the coupling is zero. For smaller systems or other real
lattices supply `tmat` directly. Projection additionally requires hopping
invariant under one-site translation.

### General integral model and molecular adapter

`ElectronPhononModel(h1e, eri, omega, coupling, nelec, ecore=0.0)` owns
real, finite C-order `float64` copies in a fixed orthonormal electronic basis.
`norb>0` and `nmode>=0` are independent. `nelec` is an explicit integer tuple
`(N_alpha,N_beta)`, with each count in `[0,norb]`.

| Attribute | Shape / meaning |
|---|---|
| `h1e` | `(norb,norb)`, real symmetric one-electron integrals |
| `eri` | `(norb,norb,norb,norb)`, full chemists' `(pq\|rs)` tensor |
| `omega` | `(nmode,)`, positive frequencies in Hartree |
| `coupling` | `(nmode,norb,norb)`, real symmetric `g[x,p,q]` in Hartree |
| `ecore` | Separate finite scalar constant, in Hartree |
| `norb`, `nmode` | Electronic orbital count and phonon mode count |

`eri` multiplies `1/2 sum_{pqrs,spin,spin'} a_p† a_r† a_s a_q` in the
Hamiltonian; its full values are retained without assuming eight-fold
Coulomb symmetry. `coupling[x,p,q]` multiplies `a_p† a_q (b_x+b_x†)`.
There is no density centering, phonon zero-point energy, or implicit mass
or frequency factor. Zero modes use `omega.shape=(0,)` and
`coupling.shape=(0,norb,norb)`. The `h1e` and coupling symmetry tolerance is
`1e-12` absolute; only roundoff within this tolerance is symmetrized.

`MolecularEP(mol, *, omega, coupling_ao)` inherits this integral interface.
It requires an already-built PySCF `gto.Mole` and does not run SCF.
`coupling_ao.shape=(nmode,nao,nao)` contains real symmetric AO matrix elements
of the coefficients of `b_x+b_x†`, in Hartree. Supply already normalized
couplings, not raw coordinate derivatives.

The adapter directly uses `mol.nelec`, `mol.energy_nuc()`,
`mol.intor_symmetric('int1e_ovlp')`, `scf.hf.get_hcore(mol)`,
`lo.orth.lowdin(S)` and `ao2mo.kernel(mol,X,compact=False)`. It retains all
AOs and raises if the overlap minimum/maximum eigenvalue ratio is not
greater than `1e-10`. `ecore` is the nuclear repulsion, to be added once to
electronic/bosonic energies. No frozen-core or active-space reduction is made.

Additional attributes are `mol` (the original object by reference),
`overlap_ao=(nao,nao)`, `ao_to_orth=(nao,norb)` and
`orthogonalization='lowdin'`. The map `X=ao_to_orth` obeys
`X.T @ overlap_ao @ X = I`, `C_AO=X @ C_orth`, and
`D_AO=X @ D_orth @ X.T`. The inverse coefficient map is `X.T @ overlap_ao`.
Integral arrays and the metric are construction-time snapshots; recreate
the model after modifying the molecule. Symmetric Löwdin differs from the
paper's meta-Löwdin basis, so an approximate LF result should only be compared
with a paper calculation after matching the basis convention.

`HubbardHolstein.to_general()` returns an independent `ElectronPhononModel`
with `h1e=tmat`, `eri[p,p,p,p]=U`, `coupling[x,p,q]=g*delta[x,p]*delta[p,q]`,
`omega=full(L,omega)` and `ecore=0`. The factor `1/2` and spin sums yield
exactly the original Hubbard double-occupation interaction.

`ExactED`, `LFHF` and `LFMP2` accept these integral models. Their general
paths require `norb<64` and pair-exchange/real-Hermitian ERIs,
`V[p,q,r,s]=V[r,s,p,q]=V[q,p,s,r]` (absolute tolerance `1e-12`). They retain
full tensors; individual-pair Coulomb symmetry is not assumed after LF
transformation. CS, MR and MRPT2 still require `HubbardHolstein`.

## Mean-field references

`CSHF(model, *, gtol=1e-8, max_cycle=1000)`:

- `kernel(mo_coeff0=None)` accepts `(2,L,L)` orbitals, occupied columns first.
- Default orbitals are hopping eigenvectors. One electron uses the original
  density fixed-point algorithm; other sectors use occupied/virtual BFGS.
- Attributes: `mo_coeff=(2,L,L)`, `mo_energy=(2,L)`, `shift=(L,)`.
- Results include `spin_density=(2,L)`, independent `orbital_grad_norm`,
  `shift_residual`, and method-specific convergence/iteration fields.
- Multi-electron BFGS also reports `optimizer_success`, `residual_ok` and
  `energy_consistent`. Physical convergence requires the commutator residual
  <= `max(1e-7,10*gtol)` and an energy check <= `1e-9`.

`LFHF(model, *, gtol=None, max_cycle=None)`:

- `kernel(mo_coeff0=None, lam0=None)` optimizes the full `(L,L)` LF matrix.
- Default `lam0[x,p]=g/(omega*L)` and `shift=0`.
- One-electron defaults are `gtol=1e-8`, `max_cycle=500`; multi-electron
  defaults are `1e-6`, `1000`, preserving the existing optimizers.
- Attributes add `lam=(L,L)`; results retain optimizer fields and existing
  stationarity diagnostics. The single-electron active-spin orbital spectrum
  is the effective one-body spectrum used by its original LF-MP2 partition.
  Empty-spin spectra have no role in that calculation.

For `ElectronPhononModel`/`MolecularEP`, LF-HF instead uses:

- `mo_coeff0=(2,norb,norb)`, orthogonal, occupied columns first;
  default orbitals come from an ordinary PySCF UHF calculation.
- `lam0=(nmode,norb)` with no locality constraint. An omitted array uses
  the ordinary density's mode force divided by `omega*N_e`, replicated
  across orbitals. Zero modes are supported.
- Joint occupied/virtual rotations and LF optimization by numerical-gradient
  BFGS, evaluating energies and Fock matrices with PySCF UHF/J/K.
- `shift=0` gauge. At fixed nonzero electron number a uniform lambda row
  shift exactly represents a coherent shift; no variational freedom is lost.
- Attributes `mo_coeff=(2,norb,norb)`, `mo_energy=(2,norb)`,
  `lam=(nmode,norb)`, `shift=(nmode,)`. Occupied and virtual blocks are
  separately canonicalized. All spin sectors use the UHF Fock partition.
- Independent `orbital_grad_norm` and `lam_grad_max` must be at most
  `max(1e-7,10*gtol)`, with canonicalization energy change below `1e-9`.
  `optimizer_success` is recorded separately from physical convergence.
- `density` is the transformed determinant density; `physical_density`
  includes LF overlap factors. Their diagonals agree. `spin_square` and
  `phonon_mean=<b>=-lam @ total_density_diagonal` are reported.

`site_density()` returns a copy of the `(2,L)` spin-resolved site density
for a converged HF object. Both drivers optimize one start. Multi-start
branch selection is explicit in `examples.scan`, allowing callers to choose
and audit their own reference recipe.

For general models, `site_density()` means occupation of the fixed
orthonormal basis and has shape `(2,norb)`; it is not an atomic population
analysis. `examples.molecular_symmetry` records six starts, the constrained
symmetric stationary branch and its broken partners for a small H2 example.

## Single-reference MP2

`CSMP2(hf, *, denominator_tol=1e-10)` uses a converged `CSHF` reference.
`kernel()` adds mixed electronic-single/one-phonon and opposite-spin Hubbard
double corrections. Results include `hf_energy`, `total_energy`,
`mp2_correction`, separate channel corrections and `minimum_denominator`.
The analytic CS expression only needs one phonon. Its low-level function
checks canonical Fock energies, orbital/shift stationarity and positive
excitation denominators. Orbital stationarity is checked at `1e-7`, so
loosening HF tolerance does not automatically relax MP2's reference check.

`LFMP2(hf, *, max_total=10, max_excited_modes=None, denominator_tol=1e-10)`
uses a converged `LFHF` reference. Single-electron calculations preserve
the original all-mode effective-one-body partition and ignore the mode-count
restriction, recording `max_excited_modes=L`. Their existing low-level
positive-denominator checks apply. Multi-electron calculations use the
unrestricted Fock partition and the requested mode-count restriction.
The chosen reference energy is checked against the supplied HF result.
`None` preserves the lattice default of two modes, while general models
default to all modes.

For general models the first implementation explicitly forms the electronic
Hamiltonian and coupling matrices, limited to 256 electronic determinants.
Analytic vacuum-to-number oscillator matrix elements leave LF displacement
untruncated. Every configuration up to `max_total` and `max_excited_modes`
includes electronic ranks 0, 1 and 2; the vacuum includes ranks 1 and 2.
Results add `phonon_double_correction`, `electronic_dimension`,
`reference_energy_error`, `higher_rank_source_error`, and `partition`.
All electron sectors use canonical UHF Fock denominators, including one
electron; this differs from the archived lattice one-electron partition.
Invalid or near-zero excitation denominators raise. No symmetry compression
of the transformed two-electron tensor is used.

The multi-electron LF routine adds the phonon scalar `omega*shift[x]`
once per one-phonon configuration. This supports nonzero coherent shifts
at the low-level fixed-reference boundary and gives the correct CS limit.
The optimized LF-HF references use zero shift, preserving archived results.
`converged=True` reports a successful finite-cutoff correction;
`result['cutoff_converged']` remains false.

## Exact ED

`ExactED(model, nmax, *, tol=1e-12, tol_residual=1e-10, max_cycle=100,
max_space=20, lindep=1e-14, coupling_convention='paper')` solves by matrix-free Davidson.
`nmax>=0` is the per-mode cutoff. `kernel()` stores structured
`coeff.shape=(d_alpha,d_beta)+(nmax+1,)*L`, energy, dimension and independently
evaluated residual. `site_density()` contracts the probability density
without constructing an RDM.

`coupling_convention` is `'paper'` or `'centered'`. The returned energy is
in the convention solved. `converged` is based on the independent residual;
cutoff convergence requires separate calculations. The Davidson `lindep` threshold can stop expansion before a very tight
residual target is met; the independent residual determines success. The
scan examples explicitly use `lindep=1e-22` for their tight ED solves.
The old low-level
`exact.kernel` still returns `(energy, structured_coeff, residual)`.

For general models, `coeff.shape=(d_alpha,d_beta)+(nmax+1,)*nmode` and
`result` also records `norb`, `nmode`, `nelec`, and `ecore`. The scalar offset
is included once. PySCF `direct_nosym` handles full electronic tensors;
independent Slater–Condon diagonals precondition Davidson. No electronic or
full electron/phonon dense matrix is formed. General models accept only
`coupling_convention='paper'`, denoting their supplied uncentered coupling.
The initial vector has reproducible full support (seed 417), covering
disconnected blocks; the PySCF diagonal preconditioner uses a 0.05 Ha level
shift. The final vector is normalized before its independent residual is
evaluated. The zero-electron sector returns the exact bare-oscillator vacuum.

## LF-MR-NOCI

`LFMR(model, lam, shift=None, *, character=None, overlap_cut=1e-10,
condition_limit=1e8)`:

- Inputs: `lam=(K,L,L)` indexed `[frame,mode,site]`, `shift=(K,L)`.
  Omitted shift is zero. All site determinants are retained initially.
- `character=None` treats the frames as the actual unprojected list.
  `+1/-1` treats them as orbit seeds projected to momentum zero/pi.
- `kernel()` minimizes CI coefficients at fixed supplied frame parameters.
- `result` includes `energy`, `rank`, structured `coeff`, flattened `hmat`,
  `smat`, `norm_error`, retained residual, reference spectrum/gap and
  `near_degenerate` (gap <= `1e-8`). Projected results also include
  `expanded_coeff` and the orbit's `lam/shift`.
- Object `lam/shift` always remain the supplied seeds; orbit arrays in
  `result` are separate. `site_density()` evaluates the physical orbit state.
- `condition_limit` chooses between direct generalized and canonical solves;
  it does not change retained rank. Projected calculation uses the same
  solver policy. `converged` requires norm error < `1e-9` and retained
  residual < `1e-10`.

## LF-MRPT2

`LFMRPT2(mr, max_total, *, external_cut=1e-10, precision_mode='float64')`
uses a converged MR object and its frame seeds, symmetry sector and
reference overlap cutoff. `max_total>=0` is an inclusive all-mode relative
phonon cutoff; zero gives empty external space.

`kernel()` assembles six blocks, removes the entire retained reference span,
orthogonalizes external candidates, transforms Hamiltonian blocks and solves
the block-Hamiltonian response. It checks that its reference energy agrees
with the supplied MR energy. Numerical failures raise exceptions without
changing cutoffs or denominators.

Results include:

| Field | Shape / meaning |
|---|---|
| `reference_energy`, `second_order_energy`, `corrected_energy` | $E_0,E^{(2)},E_0+E^{(2)}$ |
| `reference_coeff` | `(r,)` retained reference ground-root coefficients |
| `first_order_coeff` | `(s,)` retained orthonormal external response |
| `raw_reference_coeff` | `(K*d_e,)` physical reference expansion |
| `raw_first_order_ref_coeff` | `(K*d_e,)` reference part of orthogonalized response |
| `raw_first_order_candidate_coeff` | `(K*d_e*N,)` candidate part of response |
| `reference_rank`, `external_rank` | Retained dimensions |
| `reference_overlap_eigenvalues`, `external_overlap_eigenvalues` | Original canonical Gram spectra |
| `reference_eigenvalues`, `external_eigenvalues`, `denominators` | Hamiltonian spectra and external gaps |
| `reference_residual`, `response_residual` | Independent equation checks |
| `reference_metric_error`, `external_metric_error`, `cross_metric_error` | Physical retained-space Gram checks |
| `first_order_norm_squared`, `minimum_denominator` | Perturbative diagnostics |
| `max_total`, `occupations`, `reference_cut`, `external_cut`, `character` | Space definition |
| `precision_mode`, optional `precision_diagnostics` | Arithmetic and retained-space repair |

`precision_mode` is `'float64'` or `'reconditioned_longdouble'`. The latter
requires extra accumulator precision and preserves selected ranks/spectra.
Both raw kernels and final solver arrays remain `float64`.
`converged` means numerical validity at this supplied cutoff; `cutoff_converged`
is false. Nonvariational corrected energies can lie below ED.

## Low-level access and failure behavior

Examples:

```python
from mr_lf.basis import make_electron_basis, make_shape
from mr_lf.phonon import displacement_number_matrix
from mr_lf.mr.kernels import build_projected_mr_candidate_blocks
from mr_lf.mr.space import orthogonalize_mr_spaces_blockwise
from mr_lf.mr.pt2 import solve_mrpt2_blocks, mrpt2_from_blocks
```

See function docstrings for strict shapes/dtypes. Wrong model/array contracts
raise `TypeError` or `ValueError`; unsafe gaps/denominators raise `ValueError`;
linear-algebra failures propagate as `numpy.linalg.LinAlgError`; unavailable
extended precision raises `RuntimeError`. Post-reference calculations refuse
unconverged references. Example scans preserve method failures in their CSV
and exit unsuccessfully rather than labeling missing results valid.

All calculation objects are stateful and should be recreated for a changed
model or reference. Frame and input orbital arrays are copied; kernels do not
modify caller inputs. No class changes global PySCF thread settings. Example
entry points explicitly use one PySCF thread.
