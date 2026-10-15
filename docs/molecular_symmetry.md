# Molecular point groups, active CI rotations, and the cavity Hamiltonian

This guide describes molecular group data, active electronic CI rotations,
dipole-derived cavity integrals, and the implemented joint electron/photon
LFMR projection and fixed-pool reference selection. PySCF supplies electronic
integrals and CI rotations; MolSym supplies finite-group operations and
irreducible representations. The molecular LFMR path reuses the existing
coherent-state matrix elements and nonorthogonal CI construction.

## 1. Basis choices and software responsibilities

| Layer | Choice | Purpose |
|---|---|---|
| Molecular AO basis | STO-3G in the examples; all AOs retained | Keep independent ED checks small |
| Fixed orthonormal basis | Symmetric Lowdin, $X=S^{-1/2}$ | Define electronic operators, CI, and the density-diagonal LF generator |
| Canonical MOs | Optimized spin-dependent LF-HF orbitals | Define occupied/virtual states and LF-MP2 Fock denominators |

The orbital and photon mode counts are independent. No frozen-core or
active-space approximation is introduced. The minimum/maximum AO overlap
eigenvalue ratio must exceed $10^{-10}$; near-dependent AOs are rejected.
Symmetric Lowdin differs from the paper's **meta-Lowdin** choice. Complete
electronic CI energies are invariant under orthogonal changes of basis,
whereas a density-diagonal LF approximation depends on its fixed orbital
basis. These small examples are method checks, not a reproduction of the
paper's complete numerical setup.
No additional CCSD(T)-minus-MP2 electronic correction from the paper's
Eq. (42) is added; the example reports the direct LF-MP2 result.

| Software | Reused functionality |
|---|---|
| PySCF 2.14.0 | AO integrals, AO2MO, Lowdin, real spherical harmonics, cistring, CI rotations, UHF/J/K, Davidson |
| MolSym 1.2.0, optional | Finite-group detection, operations, multiplication tables, irreducible representation matrices |
| NumPy / SciPy | Tensor algebra, optimization, linear algebra and independent small-space checks |
| POLAR / QSym² | Formula/algorithm references; neither is a runtime dependency or a replacement LFMR solver |

The recorded environment uses Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1.
From the standalone `si/` source directory, install with:

```bash
python -m pip install ".[symmetry]"
```

Core installation with `python -m pip install .` suffices for `MolecularPF`,
the existing solvers, and active CI rotations. MolSym is imported lazily
only when generating group data.

## 2. Active electronic rotations

An active spatial operation acts around origin $o$ as

$$
[R(g)\phi_p](r)=\phi_p\bigl(O_g^{-1}(r-o)+o\bigr)
=\sum_q\phi_q(r)U_g[q,p].
$$

Column $p$ therefore contains the rotated input orbital $p$. The AO
coefficient action preserves the AO metric, and its orthonormal form is

$$
U_{AO}^{T}S U_{AO}=S,\quad X^TSX=I,\quad
U_{orth}=X^{-1}U_{AO}X=(X^TS)U_{AO}X.
$$

Use `orbital_actions[g]` for the CI transformation, not a three-dimensional
spatial matrix or an action in nonorthogonal AOs. In a fixed electron-number
sector, expanding the ordered creation operators gives

$$
M_\sigma(U)[J,I]=\det U[J_{occ},I_{occ}],\qquad
C'=M_\alpha(U)C M_\beta(U)^T.
$$

The minors contain the fermion reordering signs. For example, the cycle
$0\to1\to2\to0$ maps occupied orbitals $\{0,2\}$ to $\{1,0\}$, with a minus
sign when the result is reordered to $\{0,1\}$.
PySCF's orbital-rotation interface is a **passive change of basis**; the
active wrapper therefore passes `U.T`. CI may be complex while U is real.
There is no conjugation of CI and no extra normalization.

`ci` is a finite `float64` or `complex128` array of shape
`(comb(norb,N_alpha), comb(norb,N_beta))`. Both axes use PySCF `cistring`
determinant addresses, distinct from bit-string integers and MO indices.
Incorrect shapes are rejected before PySCF can silently reshape them.
The independent minor oracle also verifies

$$
\mathcal R(U)\mathcal R(V)C=\mathcal R(UV)C,\quad
\mathcal R(U^T)\mathcal R(U)C=C,\quad
\langle\mathcal R(U)C_1|\mathcal R(U)C_2\rangle=\langle C_1|C_2\rangle.
$$

### Runnable electronic example

```python
import numpy as np
from pyscf import gto
from mr_lf import molecular_point_group, apply_electronic_symmetry

a = 0.9  # Angstrom
mol = gto.M(
    atom=[("H", (0, 0, 0)), ("H", (a, 0, 0)),
          ("H", (a / 2, np.sqrt(3) * a / 2, 0))],
    basis="sto-3g", unit="Angstrom", charge=1, spin=0, verbose=0,
)
adapted = molecular_point_group(mol)  # Bare-molecule D3h
ci = np.zeros((3, 3), dtype=np.complex128)
ci[0, 0] = 1.0
g = next(i for i, u in enumerate(adapted.orbital_actions)
         if not np.allclose(u, u.T))
rotated = apply_electronic_symmetry(ci, mol.nao_nr(), mol.nelec,
                                   adapted.orbital_actions[g])
print(adapted.group.name, rotated.shape, np.linalg.norm(rotated))
```

The nonsymmetric operation distinguishes active from passive action;
an H2 site swap alone cannot detect a transposition mistake.
`molecular_point_group` runs no SCF and does not modify the geometry.
Operations are expressed in the original laboratory frame;
`atom_map[g,i]` is atom i's destination. Equivalent atoms need identical AO
bases. AO adaptation currently supports real spherical harmonics, including
contracted p/d shells. Group data retain complex characters and all partner
components. For continuous linear-molecule groups, supply a finite subgroup
explicitly, e.g. `subgroup="D2h"` for H2.

## 3. Dipoles and light-matter coupling

The model uses fixed nuclei, the long-wavelength approximation and the
dipole gauge. Photon frequencies $\omega_x$ and cavity strengths $\lambda_x$
are inputs; no molecular vibrational frequencies or coupling derivatives
are calculated. Energies are in Hartree. Explicit `origin` is always in
Bohr, independently of the units used for the input nuclear coordinates.

Following the paper's sign convention, define electron position minus
nuclear position dipole:

$$
\hat D_k=\sum_{pq\sigma}r^k_{pq}a^\dagger_{p\sigma}a_{q\sigma}
-d^{nuc}_k\hat 1,\quad
d^{nuc}_k=\sum_A Z_A(R_{A,k}-o_k),\quad
r^k=X^T\langle AO|r_k-o_k|AO\rangle X.
$$

This is the negative of the usual molecular charge dipole, so the coupling
sign must be kept consistent. The default origin is the nuclear-charge
center, $o=\sum_A Z_A R_A/\sum_A Z_A$, where $d^{nuc}$ vanishes to numerical
precision. At a custom origin, the nuclear dipole must still be subtracted
**once**, not once per electron.

To reuse a purely one-body electronic coupling interface in a fixed
$N_e>0$ sector, distribute the constant as

$$
\mu^k_{pq}=r^k_{pq}-\frac{d^{nuc}_k}{N_e}\delta_{pq},\qquad
\sum_{pq\sigma}\mu^k_{pq}a^\dagger_{p\sigma}a_{q\sigma}=\hat D_k,
$$

using $\hat N=N_e\hat 1$. Do not reuse this model after changing its electron
number. Zero-electron sectors and ECP/pseudopotential nuclear-charge
conventions are explicitly unsupported by this adapter.

For each unit polarization vector $e_x$, let $\mu_x=\sum_k e_{xk}\mu^k$.
The coefficient multiplying $a_p^\dagger a_q(b_x+b_x^\dagger)$ is

$$
g^x_{pq}=-\sqrt{\frac{\omega_x}{2}}\lambda_x\mu^x_{pq}.
$$

Polarizations are validated rather than silently normalized. No additional
frequency or mass factor should be applied by the caller. The lattice
parameter $\alpha=g^2/\omega$ is not used to label molecular cavity strengths.

## 4. Dipole self-energy and finite-basis convention

Cui Eqs. (7)-(9) use the square of the **second-quantized, projected dipole**:

$$
\hat H_{DSE}=\frac12\sum_x\lambda_x^2\hat D_x^2.
$$

Normal ordering with the fermion anticommutation relation gives

$$
\hat D_x^2=
\sum_{pq\sigma}(\mu_x\mu_x)_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\sum_{pqrs\sigma\tau}\mu^x_{pq}\mu^x_{rs}
a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}.
$$

The Hamiltonian already places a factor 1/2 before its chemists' ERI term.
Consequently the modified integrals are

$$
h_{pq}=h^{bare}_{pq}+\frac12\sum_x\lambda_x^2(\mu_x\mu_x)_{pq},\qquad
V_{pqrs}=V^{bare}_{pqrs}+\sum_x\lambda_x^2\mu^x_{pq}\mu^x_{rs}.
$$

Matrix products are formed in the orthonormal basis. Multiplying bare AO
matrices without their metric would be incorrect. This convention differs
from quadrupole integrals $\langle\phi_p|(e\cdot r)^2|\phi_q\rangle$ in a
finite orbital basis. The Eq. (10) correction is not included;
`dse_convention="projected_dipole_product"` records this choice. Modes with
different polarizations or strengths are summed individually.

The distributed nuclear constant is already included in these one/two-body
integrals. Do not add a second nuclear-dipole-square constant. `ecore`
remains nuclear repulsion alone and is included once. There is no photon
zero-point constant. Completing the square provides a useful sign check:

$$
\omega_x b_x^\dagger b_x
-\sqrt{\frac{\omega_x}{2}}\lambda_x\hat D_x(b_x+b_x^\dagger)
+\frac{\lambda_x^2}{2}\hat D_x^2
=\omega_x\left(b_x^\dagger-\frac{\lambda_x}{\sqrt{2\omega_x}}\hat D_x\right)
\left(b_x-\frac{\lambda_x}{\sqrt{2\omega_x}}\hat D_x\right).
$$

Replacing the old artificial odd-mode coupling by dipole integrals while
omitting DSE would therefore define a different Hamiltonian. No lattice
density-centering energy correction is applied to this model.

## 5. Public solver usage and convergence

```python
from pyscf import gto
from mr_lf import MolecularPF, ExactED, LFHF, LFMP2

mol = gto.M(atom="H 0 0 -0.37; H 0 0 0.37", unit="Angstrom",
            basis="sto-3g", charge=0, spin=0, verbose=0)
model = MolecularPF(mol, omega=[0.466751], polarization=[[0., 0., 1.]],
                    lambda_c=[0.05])
ed = ExactED(model, nmax=8, max_space=80, max_cycle=600, lindep=1e-22)
ed.kernel()
hf = LFHF(model, gtol=1e-8, max_cycle=1000)
hf.kernel()  # One initial guess; no global-minimum claim
if hf.converged:
    pt = LFMP2(hf, max_total=8)
    pt.kernel()
    print(ed.e_tot, ed.result["residual"], hf.e_tot, pt.e_tot)
```

This short example demonstrates a single solve. `ed.converged` certifies
the supplied truncated eigenproblem, not photon-cutoff convergence.
Neither `nmax=8` nor one `max_total=8` calculation alone establishes a
converged result. LF-MP2 is not variational; an energy below ED is not
automatically more accurate.

| Field | Meaning |
|---|---|
| `model.h1e`, `eri` | Orthonormal integrals **already including DSE** |
| `model.coupling`, `omega` | `float64 (M,norb,norb)` and `(M,)` |
| `model.origin`, `nuclear_dipole` | `(3,)`, dipole origin in Bohr and nuclear position dipole |
| `model.dipole_orth` | `(3,norb,norb)`, effective Cartesian dipoles including nuclear distribution |
| `model.projected_dipole` | `(M,norb,norb)`, polarization-contracted effective dipoles |
| `model.h1e_dse` | One-body DSE already present in h1e, exposed for inspection |
| `ed.result['coeff']` | Structured `(nstra,nstrb)+(nmax+1,)*M` electron/Fock tensor |
| `hf.lam`, `hf.shift` | `(M,norb)`, `(M,)` LF frame; the optimizer uses zero-shift gauge |

Construction performs no SCF and does not modify the input Mole. The model
is an integral snapshot: rebuild after changing geometry, basis or electron
number. Other solver definitions remain those in [theory](theory.md) and
[API](API.md).

### Complete command with saved convergence history

From the [research checkout](https://github.com/SunsetStand/my_lf_mrci),
with this SI package installed:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.si.molecular_pf --output outputs/molecular_pf_demo
```

Defaults are H2/STO-3G, R=0.74 Angstrom, one z-polarized mode,
$\omega=0.466751$ Ha and $\lambda_c=0.05$ au. They are explicit method-test
settings, not a complete reproduction of the paper's basis and mode setup.
The CLI saves one deterministic LF-HF seed: h1e eigenvectors for both spins,
$\lambda_{xp}^{initial}=g^x_{pp}/\omega_x$ and $z=0$. This explicit recipe
differs from omitting initial guesses in the minimal API example above;
both obtain one local stationary solution.

ED tries `Nmax=2,4,8,12,16`; LF-MP2 separately tries
`max_total=2,4,8,12,16`. Each cutoff check requires agreement below
$10^{-9}$ Ha between adjacent requested converged calculations. ED also
requires an independently evaluated residual below $10^{-10}$. HF or MP2
failure and unconverged cutoffs remain explicit in the output, and the CLI
returns a nonzero exit code. Existing output directories are not overwritten.

`summary.json` records conventions, parameters, basis, initialization,
dimensions, cutoff energies/residuals, versions and source hashes.
`inputs.npz` saves the actual integral data and obtained states/frames.
Use `--lambda-c 0.5` for the second strong-coupling anchor; it must satisfy
its own cutoff checks.

## 6. Find the symmetry of the whole cavity Hamiltonian

The bare molecule's group is a candidate parent group, not automatically the
symmetry of a chosen cavity. A spatial operation $O_g$ has an electronic
orbital action $U_g$ and a photon-mode action $Q_g$. With
$G_x=\sum_{pq\sigma}g^x_{pq}a_p^\dagger a_q$, joint invariance requires

$$
U_g h U_g^T=h,\qquad
V_{pqrs}=\sum_{abcd}U_{g,pa}U_{g,qb}U_{g,rc}U_{g,sd}V_{abcd},
$$

$$
Q_g\Omega=\Omega Q_g,\qquad
U_g g^x U_g^T=\sum_y Q_g[y,x]g^y,
\qquad \Omega=\operatorname{diag}(\omega).
$$

The last equation follows by transforming both factors in
$\sum_x G_x(b_x+b_x^\dagger)$. It fixes the orientation of Q: an active
photon coherent amplitude transforms as $\eta\mapsto Q_g\eta$. Frequencies
that differ cannot be mixed by a symmetry operation.

`joint_symmetry(model, group, orbital_actions, photon_actions)` validates
these conditions, orthogonality, and both representation multiplication
laws. It returns a frozen `JointSymmetry` with owned, read-only
`float64 (G,norb,norb)` orbital actions and `(G,M,M)` photon actions.
The group also owns its operation table and `complex128` irrep matrices.
Use this factory for synthetic models with explicitly known actions.

`molecular_cavity_symmetry(model, subgroup=None, tolerance=1e-9)` constructs
them automatically for `MolecularPF`. Linear molecules still require an
explicit finite parent subgroup. In each frequency block define the
$3\times M_b$ matrix with columns $A_x=\lambda_x e_x$. If
$O_g A A^T O_g^T=A A^T$, the implementation uses

$$
Q_g=A^+O_g A+I-A^+A.
$$

Here $A^+$ is the Moore-Penrose inverse. The first term rotates the coupled
mode directions; the second leaves uncoupled combinations unchanged.
Thus two equal-frequency, equal-strength in-plane modes can preserve D3h
even though each polarization alone does not. Unequal strengths or
frequencies can reduce the surviving subgroup. Zero-coupling modes have
the identity action.

Operations must also preserve the actual h, ERI and coupling matrices.
The retained operations must close under multiplication; ambiguous
tolerance-dependent, nonclosed selections raise. MolSym identifies the
subgroup and supplies its canonical operation/irrep order, while spatial
operations remain in the laboratory frame. The returned group may
therefore have different irrep labels from its parent.

For a neutral molecule the total dipole is independent of a common
electronic/nuclear origin shift. A charged molecule's dipole acquires a
constant under that shift. A joint operation may then require a photon
translation as well as an orthogonal rotation. This implementation does
not include photon translations: actual-integral filtering reduces the
group, or rejects an inconsistent representation. The default
nuclear-charge-center origin is the initial choice for the tests.

## 7. Relative kernels and symmetry restoration

A density-diagonal LF basis state also includes a conditional photon displacement:

$$
|A,I\rangle=|I\rangle|\eta_{AI}\rangle,\qquad
\eta_{AI,x}=z_{A,x}-\sum_p\lambda_{A,xp}n_p(I).
$$

Here $I$ is an alpha/beta determinant pair in the fixed Lowdin orbital
basis, and $n_p(I)$ includes both spins. A candidate frame A supplies all
allowed electronic CI coefficients within that frame, not just one
optimized HF determinant. `lam` and `shift` have shapes `(F,M,norb)` and
`(F,M)`; all frame parameters are real `float64`.

An orbital-mixing operation generally cannot be represented by permuting
the diagonal LF parameters. Instead, let $T_g$ be its determinant action,
$a=\eta_{AI}$ and $b=Q_g\eta_{BJ}$. Coherent-state identities give

$$
s(a,b)=\langle a|b\rangle=e^{-\|a-b\|^2/2},\qquad
\langle a|b_x+b_x^\dagger|b\rangle=(a_x+b_x)s(a,b),
$$

$$
\langle a|b_x^\dagger b_x|b\rangle=a_xb_xs(a,b).
$$

The electronic and photon factors combine into relative kernels

$$
K^S_{AI,BJ}(g)=s(a,b)(T_g)_{IJ},
$$

$$
K^H_{AI,BJ}(g)=s(a,b)\left[
(H_{el}T_g)_{IJ}+\sum_x(a_x+b_x)(G_xT_g)_{IJ}
+\left(E_{core}+\sum_x\omega_xa_xb_x\right)(T_g)_{IJ}\right].
$$

$H_{el}$ excludes the scalar nuclear repulsion, so $E_{core}$ enters
once. `lf_group_relative_blocks` yields one such `(F,D,F,D)` pair per
operation. These matrices are generally not individually Hermitian.
The coherent-state formulas use the full oscillator space, with no
photon occupation cutoff and no requirement that a rotated state have a
new density-diagonal LF frame.

For an irrep $\Gamma$ of dimension $d$, define matrix-unit projectors

$$
P^\Gamma_{mn}=\frac{d}{|G|}\sum_g D^\Gamma_{mn}(g)^*R(g),\qquad
P_{mn}P_{kl}=\delta_{nk}P_{ml},\quad P_{mn}^\dagger=P_{nm}.
$$

Choose one partner row m and retain every column n:
$|A,n,I;m\rangle=P_{mn}|A,I\rangle$. Since H commutes with the group,
the two-sided projected kernel reduces to a single relative sum,

$$
\langle A,n,I;m|H|B,k,J;m\rangle
=\langle AI|H P_{nk}|BJ\rangle
=\frac{d}{|G|}\sum_g D^\Gamma_{nk}(g)^*K^H_{AI,BJ}(g).
$$

The same expression with $K^S$ gives S. This avoids allocating a square
matrix over all group images. H and S are independent of the chosen row;
the row selects which partner wave function is reconstructed. Keeping
only one column would generally discard variational directions.

`lf_group_frame_blocks` returns `complex128` H/S with structured bra/ket
shape `(F,d,nstra,nstrb,F,d,nstra,nstrb)`. LFMR coefficients have shape
`(F,d,nstra,nstrb)`, with alpha before beta and PySCF determinant-address
order within each spin. Complex conjugate transposes are required even
when the input integrals are real. Flattening occurs at the NOCI solve
boundary, which solves $Hc=ESc$ after removing null/near-null metric
directions. `residual_projected` is a residual in this retained space,
not a full electron/photon Hilbert-space eigenstate residual.

## 8. LFMR usage, partner states, and selected determinants

The following example constructs two trial frames; neither is claimed to
be an optimized LFHF frame:

```python
import numpy as np
from pyscf import gto
from mr_lf import MolecularPF, LFMR, molecular_cavity_symmetry

mol = gto.M(atom="H 0 0 -0.37; H 0 0 0.37", unit="Angstrom",
            basis="sto-3g", charge=0, spin=0, verbose=0)
model = MolecularPF(mol, omega=[0.466751], polarization=[[0., 0., 1.]],
                    lambda_c=[0.05])
symmetry = molecular_cavity_symmetry(model, subgroup="D2h")
print(list(symmetry.group.irrep_matrices))  # Use these exact MolSym labels
scale = np.diagonal(model.coupling, axis1=1, axis2=2) / model.omega[:, None]
lam = np.stack([np.zeros_like(scale), scale])
shift = np.zeros((len(lam), model.nmode), dtype=np.float64)
mr = LFMR(model, lam, shift, symmetry=symmetry, irrep="A_1g", row=0)
mr.kernel()
print(mr.e_tot, mr.result["rank"], mr.result["residual_projected"])
diagnostics = mr.symmetry_diagnostics()  # Optional analytic contraction
```

Irrep names are exact backend keys, for example `A_1g` in D2h and `A_1'`
in D3h; aliases such as `Ag` are not normalized. For a multidimensional
irrep, use `0 <= row < d`. `partner_rows` lists the partners, reconstructed
with the same CI coefficients and a different projector row.

The optional `symmetry_diagnostics()` independently contracts the actual
state with relative overlap kernels. It returns

$$
r_g=\frac{\langle\Psi_m|R(g)|\Psi_m\rangle}{\langle\Psi_m|\Psi_m\rangle},
\qquad r_g=D^\Gamma_{mm}(g),\qquad
p_\Gamma=\frac{d}{|G|}\sum_g\chi_\Gamma(g)^*r_g=1.
$$

`row_transformation_error`, `representation_purity_error`, and
`norm_consistency_error` measure these separate checks. Expectations are
computed from the state rather than copied from D. A partner state need
not be invariant under every operation: only the totally symmetric
one-dimensional irrep has $r_g=1$ for all g. The optional contraction is
more expensive for large groups and is not run by `kernel()` automatically.

For a selected electronic list, pass `determinants` as an `int64 (D,2)`
array of alpha/beta **bit-string integers**, in the supplied list order.
Coefficient shape becomes `(F,d,D)`; unprojected shape is `(F,D)`.
The common selected subspace must be closed under every electronic group
action. Closure includes linear combinations and fermionic signs, not
just atom permutations. Nonclosed selections are rejected rather than
silently truncating the action. With full CI, coefficient shape without
symmetry is `(F,nstra,nstrb)`.

The current dense limits are $D\le256$ and $F d D\le1024$ for projected
calculations; unprojected calculations use $F D\le1024$. These apply
before square-kernel allocation. A group backend supporting a large
finite group does not remove these electronic/variational-space limits.
LFMR optimizes CI coefficients at fixed frames; it performs no variation
after projection, spin projection, or molecular MRPT2.
The real lattice `character=None,+1,-1` path keeps its existing behavior.
`character` and molecular `symmetry` cannot be combined; molecular
integral models use `symmetry` and `irrep` together.

## 9. Compare reference choices on one fixed pool

```python
from mr_lf import select_lf_references

results = {}
for strategy in ("unprojected", "projected", "greedy"):
    results[strategy] = select_lf_references(
        model, lam, shift, symmetry=symmetry, irrep="A_1g",
        strategy=strategy, energy_tol=1e-8, overlap_cut=1e-10,
        max_references=None,
    )
```

The selector receives a fixed pool and performs no LF/orbital optimization.
Every candidate is a frame with internal electronic CI. `unprojected`
ranks by single-frame CI energies before projection; `projected` ranks by
single-frame target-sector energies; `greedy` evaluates the actual
target-sector MR lowering of every remaining candidate at each step.
All accepted sequences are solved in the same projected sector, including
the unprojected-ranking strategy. Ties follow input frame index.

In a nonorthogonal pool, different LF parameters can generate the same
projected subspace. For retained coefficient columns B and candidate
columns C, selection first subtracts

$$
C_\perp=C-B(B^\dagger S B)^{-1}B^\dagger S C.
$$

Only the residual candidate Gram matrix is diagonalized. New directions
must exceed `overlap_cut*max(1,s_max(candidate))`, where the scale belongs
to the original individual candidate. Accepted old directions stay
unchanged. The retained spaces are therefore nested, rather than being
redefined by a changing global overlap threshold. Zero-projector and
duplicate candidates contribute no new direction. After the first
accepted frame, a gain strictly below `energy_tol` stops selection; zero
tolerance permits zero-gain new directions.

Results include `chosen_indices`, target `single_frame_energies`,
`ranking_energies`, accepted `energies`, `ranks`, `gains`, norm/residual
histories, `stop_reason`, final structured complex `coeff`, and retained
pool-coordinate basis `Q`. A zero projected pool returns no energy and
rank zero. All pool kernels are built once, so the dense limits apply to
the entire supplied pool, including candidates later rejected.

## 10. Initial calculations and independent acceptance

The initial method-test design uses H2/STO-3G at R=0.74, 1.5 and 2.0
Angstrom, one bond-polarized mode with $\omega=0.466751$ Ha, and
equilateral H3+/STO-3G at side lengths 0.9 and 1.5 Angstrom, two equal
in-plane modes with $\omega=0.5$ Ha. Both use
$\lambda_c=0,0.05,0.5$ au per mode. Nuclear repulsion is included once,
the photon zero-point constant is omitted, and the finite-basis DSE
convention remains that of Sections 3-4.

Candidate provenance must distinguish converged multistart LFHF frames
from trial perturbations of the lowest frame found. The implemented local
perturbation scales are $\pm0.5,\pm1$ times
$\operatorname{diag}(g_x)/\omega_x$, together with a zero-displacement
frame. Save every actual frame and its initial guess; the lowest found
stationary point is not a proven global minimum. Compare all three
selection strategies using exactly the same pool.

The runnable `benchmarks/si/molecular_lfmr.py` in the research repository provides `--molecule h2` or
`--molecule h3`, `--distance`, `--lambda-c`, `--output`, and `--ed-cutoffs`.
From the research repository root:

```bash
python -m benchmarks.si.molecular_lfmr --molecule h2 --distance 0.74 --lambda-c 0.05 --output outputs/h2_mr
python -m benchmarks.si.molecular_lfmr --molecule h3 --distance 1.5 --lambda-c 0.5 --output outputs/h3_mr
python -m benchmarks.si.molecular_lfmr --suite --output outputs/molecular_lfmr_suite
```

`--suite` evaluates the 15 specified parameter combinations. Each point
saves `summary.json` and `inputs_and_results.npz`; the suite additionally
saves `comparison.csv` and an incremental suite summary. Existing output
directories are rejected. The three LFHF starts use (i) core-Hamiltonian
orbitals and `diag(g)/omega`, (ii) core orbitals and zero LF parameters,
and (iii) opposite alpha/beta occupied/virtual rotations by 0.65 radians
and `diag(g)/omega`. All three initial and final arrays are saved. Only
starts passing LFHF's physical stationarity check enter the converged pool.
Local trials perturb each individual `(mode,orbital)` lambda entry of the
best found stationary frame; their shifts remain fixed. A zero frame is
always included. No stationary/global-minimum claim is attached to trials.

ED uses successive total cutoffs 2, 4, 8, 12, 16 and 24 by default and
requires both a solver residual below $10^{-10}$ and adjacent requested
energies within $10^{-9}$ Ha. LFHF uses `gtol=1e-8`, `max_cycle=1000`;
selection uses `overlap_cut=1e-10` and `energy_tol=1e-8` Ha. The saved
`complete` flag requires the recorded MR checks, at least one converged HF
start, and ED cutoff convergence. A failed check remains visible in the report.

Nearly dependent reference columns can amplify cancellation in reduced
matrix products. When canonical columns exceed 100 in magnitude, reduced
H/S products accumulate in NumPy `clongdouble` before casting to
`complex128` for LAPACK. The actual reduced metric is solved explicitly;
it is not assumed to be exactly the identity. Norms and retained residuals
are evaluated with extended accumulation on the actual saved coefficients.
This improves arithmetic without changing the supplied double-precision
kernels, the retained span, or the overlap cutoff; platform precision is
still limited by the local NumPy long-double implementation.

Coherent LFMR has no photon cutoff. An independent Fock-space ED comparison
does: solver residual and cutoff convergence must be checked separately.
Use a total-photon-number cutoff when checking symmetry with degenerate
mode mixing; a per-mode rectangular cutoff is generally not invariant
under that rotation. Save model conventions, parameters, group/irrep,
candidate provenance, retained ranks, norm errors, symmetry diagnostics,
energies and ED cutoff histories. A finite CI residual, a pure irrep, and
photon-cutoff convergence certify different properties.

### Recorded first study

The 15 specified points have been executed with these settings. Every
point passed its recorded MR, HF-pool and ED-cutoff checks. All three HF
starts converged per point; H2 pools contain 12 candidates and H3+ pools
contain 28, including duplicates that selection removes by their subspaces.
At zero coupling, the fully symmetric solution reproduces electronic FCI
within floating-point rounding.

Two stronger-coupling H3+ examples illustrate why selection strategy is
still an experimental question. Values are MR minus final finite-cutoff
ED energy, in microhartree; these include ED cutoff error.

| Side / Angstrom | lambda_c | Unprojected ranking | Projected ranking | Greedy |
|---|---:|---:|---:|---:|
| 0.9 | 0.5 | 2.060521 | 0.216338 | 0.176593 |
| 1.5 | 0.5 | 22.297512 | 22.486621 | 22.431939 |

Greedy chooses the largest next gain from the remaining pool; it does not
prove that its final accepted sequence is globally best. The strategies
stop at different frame counts and metric ranks. At 1.5 Angstrom the
remaining error is about 22 microhartree for every strategy, indicating
that this pool still leaves a physical variational error even though the
retained solve passes. No general ranking of strategies is established.

Across the recorded runs, the largest retained MR residual is
$1.972\times10^{-11}$, the largest norm error is
$1.688\times10^{-14}$, and the final representation-purity error is below
$5.552\times10^{-17}$. The largest ED solver residual is
$1.213\times10^{-13}$; the largest final adjacent-cutoff energy difference
is $9.531\times10^{-10}$ Ha. These are separate checks; a small residual
does not remove candidate-pool or photon-cutoff error.

The research checkout saves the full study under
`data/molecular_lfmr_v1/` with a comparison CSV, per-point JSON and NPZ
archives, and a result explanation. The external research workflow regenerates
the study with `--suite`; the SI library does not depend on research files.

References: [Cui, Mandal and Reichman, Eqs. (5)-(10)](https://arxiv.org/html/2310.13084v2),
[PySCF CI rotations](https://pyscf.org/_modules/pyscf/fci/addons.html#transform_ci_for_orbital_rotation),
[MolSym documentation](https://molsym.readthedocs.io/en/latest/usage.html).
