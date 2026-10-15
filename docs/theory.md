# LF-MR-NOCI and block-Hamiltonian LF-MRPT2

**Chenxi Wang & Zhihao Cui (advisor)**

This document defines the methods implemented in this repository. All MR
frame parameters are supplied and held fixed during the CI and PT2 steps.
The electronic coefficients are independent variables for every frame and
site determinant. Reference optimization, CI mixing and perturbation theory
are separate calculations.

The single-reference CS/LF methods follow the model specialization of
[Cui, Mandal and Reichman, J. Chem. Theory Comput. (2024),
DOI: 10.1021/acs.jctc.3c01166](https://doi.org/10.1021/acs.jctc.3c01166),
[arXiv:2310.13084v2](https://arxiv.org/abs/2310.13084v2).
The fixed-frame MR construction and the whole-block PT2 partition below are
additional methods defined by this implementation. The Fock partition of
single-reference LF-MP2 and the block partition of LF-MRPT2 have distinct
zeroth-order operators.

## 1. Hamiltonian, indices and representations

The local Hubbard–Holstein Hamiltonian is

$$
\hat H=\sum_{pq\sigma}t_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+U\sum_p n_{p\alpha}n_{p\beta}
+\omega\sum_x b_x^\dagger b_x
+g\sum_x n_x(b_x+b_x^\dagger),
\qquad n_p=n_{p\alpha}+n_{p\beta},\quad \alpha=g^2/\omega.
$$

There is one phonon mode per electronic site, a common positive frequency,
real symmetric hopping and real displacements. The hopping matrix is an
input; its diagonal elements are included once in the hopping contribution.
There is no phonon zero-point constant. Site indices start at zero.

| Index | Meaning |
|---|---|
| $p,q$ | Electronic site |
| $x$ | Phonon mode, associated with site $x$ |
| $\sigma=\alpha,\beta$ | Spin, stored in order 0,1 |
| $i,j$; $a,b$ | Occupied; virtual MO, within a spin sector |
| $A,B$ | LF frame, or orbit seed in a projected calculation |
| $I=(I_\alpha,I_\beta),J$ | Site determinants in a fixed electron-number sector |
| $\nu,\mu$ | Complete multimode occupation vectors |
| $k,l$ | Addresses in a supplied list of such occupation vectors |
| $r,s$ | Retained reference and external dimensions |
| $R$ | Number of lattice translations, modulo $L$ |

For $N_\sigma$ electrons, $d_\sigma=\binom{L}{N_\sigma}$ and
$d_e=d_\alpha d_\beta$. A determinant address is its position in PySCF
`cistring` order; its bit-string integer encodes occupied sites. These are
different indices. The combined determinant address is
$I_{\rm addr}=I_{\alpha,\rm addr}d_\beta+I_{\beta,\rm addr}$.
Occupied MO columns precede virtual columns; no MO index is a site address.

`float64` is used for the existing energy kernels and state coefficients;
the electronic symmetry utility in Section 13 also accepts `complex128` CI.
Configuration lists and group labels use `int64`. Array orders are:

| Quantity | Shape and order |
|---|---|
| Orbitals $C$ | `(2,L,L)`, `[spin,site,MO]` |
| MO energies | `(2,L)`, `[spin,MO]` |
| Frame $\lambda$ | `(K,L,L)`, `[frame,mode,site]` |
| Frame $z$ | `(K,L)`, `[frame,mode]` |
| MR coefficients | `(K,d_alpha,d_beta)` |
| Unprojected MR kernels | `(K,d_alpha,d_beta,K,d_alpha,d_beta)` |
| Candidate configurations | `(N,L)`, `[configuration,mode]` |
| Candidate coefficients | C order `(K,d_alpha,d_beta,N)` |
| Exact wavefunction | `(d_alpha,d_beta)+(Nmax+1,)*L` |
| Site density | `(2,L)`, `[spin,site]` |

Only linear-algebra boundaries flatten structured state indices. Reference
address is $A d_e+I_{\rm addr}$; candidate address is
$(A d_e+I_{\rm addr})N+k$. All six overlap/Hamiltonian blocks use the same
bra/ket order. Exact ED uses the site/Fock tensor; single-reference LF-MP2
uses canonical MOs.

### General molecular integral interface (adapter stage)

The integral data objects extend the Hamiltonian definition to

$$
\hat H=E_{\rm core}
+\sum_{pq\sigma}h_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\frac12\sum_{pqrs\sigma\tau}V_{pqrs}
a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}
+\sum_x\omega_x b_x^\dagger b_x
+\sum_{xpq\sigma}g^x_{pq}a^\dagger_{p\sigma}a_{q\sigma}(b_x+b_x^\dagger).
$$

Here $p,q,r,s$ label a fixed orthonormal electronic basis, distinct from
canonical MO indices, and $x$ labels an independent harmonic mode.
`h1e=(norb,norb)`, `eri=(norb,norb,norb,norb)`,
`coupling=(nmode,norb,norb)` and `omega=(nmode,)` are real `float64` arrays.
The full chemists' tensor is $V_{pqrs}=(pq|rs)$; no permutation compression
is applied by the data object. All electron counts are explicit alpha/beta
tuples. The uncentered coupling has no zero-point constant or additional
frequency/mass factors. $E_{\rm core}$ is nuclear repulsion for molecules
and zero for the lattice model.

Let $S$ be the nonorthogonal AO overlap and let the columns of $X$ define
the new orbitals $|p\rangle=\sum_\mu|\mu\rangle X_{\mu p}$.
The adapter directly calls PySCF symmetric Löwdin orthogonalization,

$$
X=S^{-1/2},\qquad X^TSX=I,\qquad
h=X^Th_{\rm AO}X,\qquad g^x=X^Tg^x_{\rm AO}X,
$$

and PySCF AO2MO for

$$
V_{pqrs}=\sum_{\mu\nu\kappa\ell}
X_{\mu p}X_{\nu q}X_{\kappa r}X_{\ell s}(\mu\nu|\kappa\ell).
$$

The smallest/largest overlap eigenvalue ratio must exceed $10^{-10}$.
Otherwise construction raises; no orbitals are deleted. All electronic
orbitals are retained, with no frozen-core or active-space approximation.
This symmetric Löwdin choice differs from the meta-Löwdin basis used in
Cui et al.; matching that convention is needed for comparing approximate
LF molecular energies with the paper.

For an orbital matrix and density matrix in the two representations,

$$
C_{\rm AO}=XC,\qquad C=X^TSC_{\rm AO},\qquad
D_{\rm AO}=XDX^T,\qquad D=X^TSD_{\rm AO}SX.
$$

Cyclicity of the trace then gives

$$
\operatorname{Tr}(g^x_{\rm AO}D_{\rm AO})
=\operatorname{Tr}(X^Tg^x_{\rm AO}XD)
=\operatorname{Tr}(g^xD).
$$

This invariance, same-state UHF energies evaluated with PySCF J/K, and
PySCF FCI energies in both bases validate the adapter's coordinate and
energy bookkeeping. They do not validate a molecular LF approximation.
The original molecule is retained by reference while all integral arrays
and the AO metric are snapshots; rebuild the model after changing the molecule.

For the lattice conversion, $h=t$, $V_{pppp}=U$, and
$g^x_{pq}=g\delta_{xp}\delta_{pq}$. The two opposite-spin terms, together
with the factor $1/2$, give $U n_{p\alpha}n_{p\beta}$; same-spin terms
vanish. This recovers the Hamiltonian above without an energy shift.

The general integral paths support ED, LF-HF/MP2 and fixed-frame LFMR.
Sections 13-14 extend these to molecular cavities, joint symmetry restoration
and fixed-pool reference selection. The lattice formulas retain their
original specialization. Molecular MRPT2, Hessians and VAP are not implemented.

### General-integral ED and density-diagonal LF reference

ED retains the structured state
`(nstr_alpha,nstr_beta)+(nmax+1,)*nmode`. PySCF `direct_nosym` applies
the electronic one/two-body operators and each general coupling matrix;
number energies and independent local oscillator ladders act on the phonon
axes. The independently derived determinant/Fock diagonal is

$$
H_{I\mathbf n,I\mathbf n}=E_{\rm core}+\sum_p h_{pp}N_p(I)
+\frac12\sum_{pq}(pp|qq)N_p(I)N_q(I)
-\frac12\sum_{pq\sigma}(pq|qp)n_{p\sigma}(I)n_{q\sigma}(I)
+\sum_x\omega_x n_x.
$$

The linear coupling has zero number-state diagonal, including off-diagonal
electronic couplings. Davidson uses this preconditioner; a separate
Hamiltonian action evaluates the final residual. Neither ED matrix is stored.
The initial vector has reproducible full support; a lowest-diagonal basis
vector alone can converge in a higher disconnected block. The PySCF
preconditioner has a finite level shift to avoid nearly dependent corrections
in diagonal problems, and the final residual uses a normalized state.

The LF ansatz in the same fixed electronic basis is

$$
|\Psi_{\rm LF}\rangle=U|\Phi_C,0\rangle,\qquad
U=\exp\left[\sum_x\left(z_x-\sum_p\lambda_{xp}n_p\right)
(b_x^\dagger-b_x)\right],\qquad \lambda\in\mathbb R^{n_{\rm mode}\times n_{\rm orb}}.
$$

Write $d_{xpq}=\lambda_{xp}-\lambda_{xq}$,
$F_{pq}=\exp[-\frac12\sum_x d_{xpq}^2]$ and $G^x_{pq}=g^x_{pq}F_{pq}$.
The exact vacuum average $\langle0|U^\dagger HU|0\rangle$ has scalar
$E_{\rm core}+\sum_x\omega_xz_x^2$ and one-body tensor

$$
\bar h_{pq}=h_{pq}F_{pq}
+\sum_xG^x_{pq}(2z_x-\lambda_{xp}-\lambda_{xq})
+\delta_{pq}\sum_x\omega_x(\lambda_{xp}^2-2z_x\lambda_{xp}).
$$

Its full chemists' two-body tensor is

$$
\begin{aligned}
\bar V_{pqrs}={}&V_{pqrs}\exp[-\tfrac12\sum_x(d_{xpq}+d_{xrs})^2]
+2\delta_{pq}\delta_{rs}\sum_x\omega_x\lambda_{xp}\lambda_{xr}\\
&-2\delta_{rs}\sum_xG^x_{pq}\lambda_{xr}
-2\delta_{pq}\sum_x\lambda_{xp}G^x_{rs}.
\end{aligned}
$$

These formulas agree with POLAR's
[`get_glf_ham_non_loc_h_ep`](https://github.com/zhcui/polar_preview/blob/master/polar/lang_firsov/lang_firsov.py)
and are independently tested against spin-bit Hamiltonians transformed
with large oscillator exponentials. The tensor retains pair exchange and
simultaneous Hermitian reversal, but generally loses separate $p\leftrightarrow q$
symmetry. It must remain full when passed to PySCF J/K or contractions.

PySCF UHF evaluates determinant energies, Fock matrices and orbital
stationarity; numerical-gradient BFGS jointly optimizes occupied/virtual
rotations and lambda. At fixed $N_e>0$ the map
$\lambda_{xp}\mapsto\lambda_{xp}+c_x$,
$z_x\mapsto z_x+N_ec_x$ leaves $U$ unchanged on the electronic sector.
Thus setting $z=0$ loses no coherent displacement freedom. This phase uses
one start per solver call, with explicit multi-start branch comparisons in
the molecular example. Independent physical gradients, rather than the
optimizer return code, determine convergence; a Hessian test is not performed.

### Finite-electronic-space general LF-MP2

For electronic determinants $I,J$ in the fixed basis define
$\eta_{Ix}=z_x-\sum_p\lambda_{xp}N_p(I)$ and
$\delta_{IJ,x}=\eta_{Jx}-\eta_{Ix}$. The infinite-oscillator overlaps are

$$
A_{IJ}(\mathbf n)=e^{-\frac12\sum_x\delta_{IJ,x}^2}
\prod_x\frac{\delta_{IJ,x}^{n_x}}{\sqrt{n_x!}}.
$$

Let $H^e_{IJ}$ exclude the scalar offset and let $G^x_{IJ}$ be the electronic
matrix of $\sum_{pq\sigma}g^x_{pq}a^\dagger_{p\sigma}a_{q\sigma}$. Then

$$
\begin{aligned}
B_{IJ}(\mathbf n)={}&H^e_{IJ}A_{IJ}(\mathbf n)\\
&+\sum_xG^x_{IJ}\left[(\eta_{Ix}+\eta_{Jx})A_{IJ}(\mathbf n)
+\sqrt{n_x}A_{IJ}(\mathbf n-\mathbf e_x)\right]\\
&+\delta_{IJ}\delta_{\mathbf n,0}
\left(E_{\rm core}+\sum_x\omega_x\eta_{Ix}^2\right)
+\delta_{IJ}\sum_x\delta_{\mathbf n,\mathbf e_x}\omega_x\eta_{Ix}.
\end{aligned}
$$

The lower-occupation term is zero when $n_x=0$. Multiplying $B$ by the
reference CI vector and rotating to canonical MOs with PySCF yields
$M_{\mu\mathbf n}=\langle\Phi_\mu,\mathbf n|U^\dagger HU|\Phi_0,0\rangle$.
The second-order correction is

$$
E^{(2)}=-\sum_{(\mu,\mathbf n)\ne(0,0)}
\frac{|M_{\mu\mathbf n}|^2}{\sum_{a\in\mu}\epsilon_a-\sum_{i\in\mu}\epsilon_i
+\boldsymbol\omega\cdot\mathbf n}.
$$

Electronic ranks 0, 1 and 2 occur for nonzero phonon occupation, including
**phonon-dressed doubles**. Vacuum excitations have ranks 1 and 2; higher
ranks are checked numerically. Excitation denominators must be positive.
The first version explicitly builds electronic matrices (at most 256
determinants), with no oscillator displacement cutoff. `max_total` truncates
the perturbative sum; `max_excited_modes` is optional and defaults to all
modes for general models. Finite-sum success does not establish cutoff
convergence or reliability near a small denominator. Every spin sector uses
the canonical UHF Fock partition, which differs from the archived
one-electron lattice partition.

### Centered coupling and cutoff conventions

A centered ED calculation replaces $n_x$ by $n_x-N_e/L$.
In the complete oscillator space, shifting every oscillator by
$gN_e/(L\omega)$ gives

$$
E_{\rm uncentered}=E_{\rm centered}
-\frac{g^2N_e^2}{L\omega}.
$$

For the one-electron four-site sector the shift is $-\alpha/4$.
A finite number-state cutoff is not invariant under this displacement.
Convergence must be checked in each representation. `ExactED` returns the
energy of the convention actually solved; it does not apply a hidden shift.
Archived centered ED tables retain the raw energy, explicit shift, residual
and cutoff evidence. Analytic MR kernels directly use uncentered coupling.

ED `nmax` is a per-mode maximum. MP2/MRPT2 `max_total=M` is inclusive:
$1\leq\sum_x\nu_x\leq M$. With all modes allowed,
$N=\binom{M+L}{L}-1$. An optional single-reference LF-MP2 mode restriction
limits the number of nonzero $\nu_x$; MRPT2 includes all modes.

## 2. LF frames and their coherent-state basis

Define the real displacement

$$
D(\boldsymbol\eta)=\exp\!\left[\sum_x\eta_x(b_x^\dagger-b_x)\right],
\qquad D^\dagger(\boldsymbol\eta)b_xD(\boldsymbol\eta)=b_x+\eta_x.
$$

A frame represents a density-conditioned transformation,

$$
\hat U_A=\exp\!\left[\sum_x\left(z_{Ax}-\sum_p\lambda_{Axp}n_p\right)
(b_x^\dagger-b_x)\right].
$$

For determinant $I$, let
$n_p(I)=n_{p\alpha}(I)+n_{p\beta}(I)$ and
$d_I=\sum_p n_{p\alpha}(I)n_{p\beta}(I)$.
The conditional displacement and vacuum basis state are

$$
\eta_{AIx}=z_{Ax}-\sum_p\lambda_{Axp}n_p(I),\qquad
|A,I\rangle=|I\rangle D(\boldsymbol\eta_{AI})|0\rangle.
$$

Expanding the electronic part of an LF-HF determinant gives a particular
linear combination of these states. MR independently varies all coefficients
$C_{AI}$, including all electronic site determinants, so it has greater
electronic flexibility than mixing one optimized HF state per frame.

For fixed $N_e>0$, the transformation
$\lambda_{Axp}\mapsto\lambda_{Axp}+c_{Ax}$,
$z_{Ax}\mapsto z_{Ax}+N_ec_{Ax}$ leaves every $\eta_{AIx}$ unchanged.
Thus the full-matrix LF-HF optimizer can choose $z=0$, using
$\lambda^{(0)}_{Axp}=\lambda_{Axp}-z_{Ax}/N_e$.
The factor $N_e$ matters for multiple electrons. The vacuum electron sector
has no such shift gauge; its physical ground state is the zero-displacement
phonon vacuum.

## 3. Vacuum overlap and Hamiltonian kernels

Baker–Campbell–Hausdorff normal ordering gives, for real displacements,

$$
|\boldsymbol\eta\rangle=
 e^{-\|\boldsymbol\eta\|^2/2}
 e^{\sum_x\eta_xb_x^\dagger}|0\rangle,
$$

and summing number-state overlaps yields

$$
G_{AI,BJ}=\langle\boldsymbol\eta_{AI}|\boldsymbol\eta_{BJ}\rangle
=\exp[-\tfrac12\|\boldsymbol\eta_{AI}-\boldsymbol\eta_{BJ}\|^2].
$$

Electronic site determinants are orthonormal, hence

$$
S_{AI,BJ}=\delta_{IJ}G_{AI,BJ}.
$$

Using $b_x|\boldsymbol\eta\rangle=\eta_x|\boldsymbol\eta\rangle$,

$$
\langle\eta|b_x|\zeta\rangle=\zeta_xG,\quad
\langle\eta|b_x^\dagger|\zeta\rangle=\eta_xG,\quad
\langle\eta|b_x^\dagger b_x|\zeta\rangle=\eta_x\zeta_xG.
$$

Let $h_{IJ}=\langle I|\sum_{pq\sigma}t_{pq}a^\dagger_{p\sigma}a_{q\sigma}|J\rangle$.
The complete kernel is

$$
H_{AI,BJ}=h_{IJ}G_{AI,BJ}
+\delta_{IJ}G_{AI,BJ}\left[Ud_I+
\omega\sum_x\eta_{AIx}\eta_{BJx}
+g\sum_x n_x(I)(\eta_{AIx}+\eta_{BJx})\right].
$$

For different electronic determinants, the full overlap is zero but the
hopping matrix element still contains the phonon Gaussian $G_{AI,BJ}$.
Electronic diagonal hopping terms remain in $h_{II}$ and are not added to
the diagonal interaction kernel.

In the implementation, `cistring` link-table entries contain creation site,
annihilation site, target address and fermionic sign. For every source
address $J_\sigma$, accumulate
$h^\sigma_{I_\sigma,J_\sigma}+=s\,t_{pq}$.
The two-spin hopping matrix is
$h=h^\alpha\otimes1_\beta+1_\alpha\otimes h^\beta$.
Site occupations and double occupancies are obtained from bit strings.
Conditional displacement products are evaluated by NumPy contractions.
These vacuum kernels are analytic in the complete phonon space and use no
ED oscillator cutoff.

### General-integral vacuum frames with an explicit CI list

For an ordered list of paired alpha/beta bit strings
$\mathcal D=\{(s_i^\alpha,s_i^\beta)\}_{i=0}^{D-1}$ in the same orthonormal
orbital basis, define $P_\mathcal D=\sum_{i\in\mathcal D}|i\rangle\langle i|$.
Different selected determinants have electronic overlap $\delta_{ij}$;
the list need not be a Cartesian product of spin-string lists. The selected
operators are $P_\mathcal D H_{\rm el}P_\mathcal D$ and
$P_\mathcal D G^xP_\mathcal D$, with $H_{\rm el}$ excluding `ecore`.

To retain the general chemists' ERI tensor without additional permutations,
write $E_{pq}=\sum_\sigma a^\dagger_{p\sigma}a_{q\sigma}$. Normal ordering gives

$$
\sum_{\sigma\tau}a^\dagger_{p\sigma}a^\dagger_{r\tau}
a_{s\tau}a_{q\sigma}=E_{pq}E_{rs}-\delta_{qr}E_{ps},
$$
$$
H_{\rm el}=\sum_{ps}\left(h_{ps}-\frac12\sum_q v_{pqqs}\right)E_{ps}
+\frac12\sum_{pqrs}v_{pqrs}E_{pq}E_{rs}.
$$

`SelectedElectronicSpace` applies this expression directly to each selected
ket. Allowed one-body actions use `cistring.cre_des_sign`, with occupation
checks before calling it; for $p=q$ the orbital must be occupied. Each spin
sector's one-body action preserves that spin's electron count, and its even
fermion parity introduces no additional sign from the other spin sector.
Only the final bra is looked up in the selected list. The intermediate state
between the two $E$ operators is unrestricted, because
$P E_{pq}E_{rs}P$ is generally different from $P E_{pq}P E_{rs}P$.
Contributions are streamed rather than storing full-space transition tables;
the dense selected matrices use the same contribution generators.

Each supplied LF frame has $|A,i\rangle=|i\rangle D(\eta_{Ai})|0\rangle$,
where $\eta_{Ai,x}=z_{A,x}-\sum_p\lambda_{A,xp}n_p(i)$ and
$f_{Ai,Bj}=\exp[-\|\eta_{Bj}-\eta_{Ai}\|^2/2]$. The general vacuum kernels are

$$
S_{Ai,Bj}=\delta_{ij}f_{Ai,Bj},\qquad
H_{Ai,Bj}=f_{Ai,Bj}\left[
h^{\rm el}_{ij}+\sum_x G^x_{ij}(\eta_{Ai,x}+\eta_{Bj,x})
+\delta_{ij}\left(E_{\rm core}+\sum_x\omega_x\eta_{Ai,x}\eta_{Bj,x}\right)
\right].
$$

This is a variational electronic truncation at fixed frames, with no PT2
correction for excluded configurations. A nested-list variational comparison
requires consistent retained physical spans: an overlap threshold can itself
remove directions. The small-space implementation keeps full ERI storage,
limits the selected list to 256 determinants and the total frame basis to
1024, and outputs `(F,D,F,D)` tensors. With no explicit list, the existing
six-axis complete-FCI representation is retained.

See the [PySCF selected-CI operator identity](https://pyscf.org/_modules/pyscf/fci/selected_ci.html)
and [cistring signs](https://pyscf.org/_modules/pyscf/fci/cistring.html).
The PySCF fast selected-CI contraction uses spin-string Cartesian-product
vectors and fourfold ERI packing; the general paired-list contraction here
retains the full ordered integral tensor instead.

## 4. NOCI coefficients and overlap truncation

For fixed frames,

$$
|\Psi_0\rangle=\sum_{AI}C_{AI}|A,I\rangle,\qquad
E[C]=\frac{C^\mathsf TH C}{C^\mathsf TS C}.
$$

Stationarity under $C^\mathsf TSC=1$ gives $HC=ESC$.
Duplicate or nearly duplicate frames make the Gram matrix singular.
Diagonalize $S=V\operatorname{diag}(s_\mu)V^\mathsf T$, retain
$s_\mu>\epsilon_R\max(1,s_{\max})$, and form

$$
X=V_{\rm keep}\operatorname{diag}(s_{\rm keep}^{-1/2}),
\quad X^\mathsf TSX=1,
\quad \bar H_R=X^\mathsf THX.
$$

Solve $\bar H_Ru_0=E_0u_0$ and recover $C=Xu_0$.
Sign changes of an eigenvector do not change physical observables.
The overlap threshold changes the retained variational space. It is not a
shift of a PT denominator. The relevant residual after truncation is
$\|X^\mathsf T(HC-E_0SC)\|$; the full raw-coordinate residual may differ.
The NOCI solver separately checks significant negative Gram eigenvalues,
symmetry; the overlap condition number selects the solver backend.

For nested physical spaces without cutoff-induced loss of directions,
adding frames cannot increase the variational lowest energy. A finite
candidate pool does not establish convergence to exact ED, and fixed-frame
MR does not reoptimize the frame parameters against the MR energy.

## 5. Translation projection and its signs

Translation acts jointly on the electronic sites and phonon modes. For a
site determinant,

$$
\hat{\mathcal T}^R|I\rangle=s_R(I)|\tau_RI\rangle,\qquad
s_R(I)=(-1)^{\sum_\sigma w_\sigma(R,I)[N_\sigma-w_\sigma(R,I)]},
$$

where $w_\sigma(R,I)=\sum_{p=L-R}^{L-1}n_{p\sigma}(I)$ counts occupied
orbitals crossing the boundary and $w_\sigma(0,I)=0$.
Reordering those orbitals to canonical site order produces the sign.
Both spin-sector signs multiply; determinant addresses are gathered through
the translated bit strings.

Translated parameters and occupations are

$$
\lambda^{(R)}_{A,xp}=\lambda_{A,(x-R)\bmod L,(p-R)\bmod L},\quad
z^{(R)}_{Ax}=z_{A,(x-R)\bmod L},\quad
\nu_x^{(R)}=\nu_{(x-R)\bmod L}.
$$

For the implemented real characters $q=+1$ and $q=-1$, define

$$
|\Phi_{AI\nu}^{(q)}\rangle=
\frac1{\sqrt L}\sum_{R=0}^{L-1}q^R\hat{\mathcal T}^R|AI\nu\rangle.
$$

$q=+1$ corresponds to momentum zero; $q=-1$ to momentum $\pi$ and requires
even $L$. The above states equal $\sqrt L$ times an idempotent symmetry
projector applied to the seeds; individual columns are not normalized and
can vanish. Their Gram matrix supplies normalization and rank selection.

For a translation-invariant operator $O$, expanding the two sums and using
$R=S+r$ gives

$$
\langle\Phi_{AI\nu}^{(q)}|O|\Phi_{BJ\mu}^{(q)}\rangle
=\sum_{r=0}^{L-1}q^r\langle AI\nu|O\hat{\mathcal T}^r|BJ\mu\rangle.
$$

The $L$ equivalent terms cancel the original $1/L$, so no extra factor
$1/L$ remains. The direct backend loops over relative translations, rolls
ket frame parameters and ket phonon occupations, gathers the translated
ket determinant, and multiplies its sign and character. A supplied
occupation list need not be translation closed. The physical ket
occupations are translated even in that case.

Projected vacuum matrices have dimension $Kd_e$, rather than $LKd_e$.
Projected candidate matrices have dimension $Kd_eN$. Exact disjoint Gram
groups are electronic translation orbits; Hamiltonian elements may couple
different groups. The direct backend avoids full-orbit square matrices.
Wavefunction projection retains coherent phases and fermionic signs;
averaging displacement parameters would define different states.

## 6. Displaced number-state matrix elements

The reference columns use $\nu=0$. External candidates are

$$
|AI\nu\rangle=|I\rangle D(\boldsymbol\eta_{AI})|\nu\rangle,
\qquad 1\leq\sum_x\nu_x\leq M.
$$

These are relative excitations in each LF frame. They are not bare site-basis
ED oscillator configurations. Candidates from different frames need not be
orthogonal to references or to each other.

### 6.1 Displacement overlaps and recurrence

For one mode let $\delta=\zeta-\eta$ and
$F_{mn}(\delta)=\langle m|D(\delta)|n\rangle$.
Real displacements have no composition phase. Normal ordering gives

$$
D(\delta)=e^{-\delta^2/2}e^{\delta b^\dagger}e^{-\delta b},
$$

and expanding the two exponentials gives the finite sum

$$
F_{mn}(\delta)=e^{-\delta^2/2}\sqrt{m!n!}
\sum_{k=0}^{\min(m,n)}
\frac{(-1)^{n-k}\delta^{m+n-2k}}{k!(m-k)!(n-k)!}.
$$

The code uses a recurrence instead of directly summing factorial terms:

$$
F_{00}=e^{-\delta^2/2},\quad
F_{m0}=\frac{\delta}{\sqrt m}F_{m-1,0},\quad
F_{0n}=-\frac{\delta}{\sqrt n}F_{0,n-1},
$$

$$
F_{m,n+1}=\frac{\sqrt m F_{m-1,n}-\delta F_{mn}}{\sqrt{n+1}}.
$$

The last relation follows from $Db^\dagger=(b^\dagger-\delta)D$.
At $\delta=0$ these elements reduce to $\delta_{mn}$, and
$F(-\delta)=F(\delta)^\mathsf T$ with exchanged rectangular limits.
A finite returned block is a submatrix of the infinite-oscillator operator.
It is not the exponential of a truncated ladder matrix and need not be
orthogonal. Large occupations/displacements require separate stability
checks; the recurrence has no uniform accuracy guarantee for all parameters.

For multiple modes, $G_{\nu\mu}=\prod_x F_{\nu_x\mu_x}(\delta_x)$.
Only the occupation rows supplied by the caller are gathered; a tensor-product
Fock space is not allocated.

### 6.2 Ladder and number operators

Using $D(\eta)^\dagger bD(\zeta)=D(\delta)(b+\zeta)$, right action on
$|n\rangle$ yields

$$
B_{mn}(\eta,\zeta)=\sqrt n F_{m,n-1}+\zeta F_{mn},
$$

$$
C_{mn}(\eta,\zeta)=\sqrt{n+1}F_{m,n+1}+\zeta F_{mn},
$$

$$
N_{mn}(\eta,\zeta)=(n+\zeta^2)F_{mn}
+\zeta\sqrt n F_{m,n-1}+\zeta\sqrt{n+1}F_{m,n+1}.
$$

Here $B,C,N$ represent $b,b^\dagger,b^\dagger b$. Terms with negative
occupations vanish. The displacement recurrence computes one auxiliary ket
column before returning the requested block, retaining the $n+1$ terms at
its boundary. Creation is the transpose of annihilation with the bra and
ket frames exchanged, rather than within one fixed frame pair.

In multiple modes, an operator in mode $x$ contributes its corresponding
$B^x,C^x,N^x$ times $\prod_{y\ne x}F^y$.
The implementation multiplies these factors directly and never divides by
the total overlap, which can be zero.

### 6.3 All overlap and Hamiltonian blocks

For any reference/candidate bra and ket, define
$\delta_x=\eta_{BJx}-\eta_{AIx}$. Then

$$
S_{AI\nu,BJ\mu}=\delta_{IJ}\prod_xF_{\nu_x\mu_x}(\delta_x),
$$

$$
\begin{aligned}
H_{AI\nu,BJ\mu}={}&h_{IJ}\prod_xF_{\nu_x\mu_x}(\delta_x)\\
&+\delta_{IJ}\Bigg[Ud_I\prod_xF^x
+\omega\sum_xN^x\prod_{y\ne x}F^y
+g\sum_xn_x(I)(B^x+C^x)\prod_{y\ne x}F^y\Bigg].
\end{aligned}
$$

Setting both occupations to vacuum recovers Section 3. Taking vacuum bra
and nonvacuum ket gives the reference/candidate block; taking two nonvacuum
lists gives the candidate/candidate block. The assembly returns
$S_{RR},S_{RT},S_{TT}$ and $H_{RR},H_{RT},H_{TT}$, with shapes
$(n_R,n_R),(n_R,n_T),(n_T,n_T)$, where $n_R=Kd_e$ and $n_T=Kd_eN$.
Hermitian reverse blocks are transposes. The vacuum is excluded from the
candidate list because it is already represented in the reference space.
Translation projection, when requested, is applied equally to both spaces
and to all six blocks before orthogonalization.

## 7. Reference projector and orthogonal external space

Let $R$ and $T$ denote formal matrices whose columns are the physical
reference and candidate states. They are never explicitly embedded in the
full Hilbert space. Their Gram matrices are the assembled $S$ blocks.
Canonical reference orthogonalization gives $\bar R=RX$ and
$\bar R^\mathsf T\bar R=1$.

The projector onto the **entire retained reference space** is
$P=\bar R\bar R^\mathsf T$; it includes its excited CI roots.
Subtracting only the chosen ground vector would define a different PT space.
Define

$$
B=X(X^\mathsf TS_{RT}),\qquad
\widetilde T=(1-P)T=T-RB,
$$

$$
G=\widetilde T^\mathsf T\widetilde T
=S_{TT}-S_{RT}^\mathsf TB.
$$

Diagonalize $G$ and retain eigenvalues
$g_\mu>\epsilon_Q\max(1,g_{\max})$.
For $Y=W_{\rm keep}\operatorname{diag}(g_{\rm keep}^{-1/2})$,

$$
Z=(T-RB)Y,\qquad Z^\mathsf TZ=1,\qquad \bar R^\mathsf TZ=0.
$$

Only retained reference directions are subtracted. Discarded reference
columns do not define extra orthogonality constraints. An empty retained
external space is valid.

The actual orthonormal Hamiltonian blocks are

$$
\bar H_R=X^\mathsf TH_{RR}X,
$$

$$
W_{RQ}=X^\mathsf T(H_{RT}-H_{RR}B)Y,
$$

$$
K=Y^\mathsf T[H_{TT}-H_{TR}B-B^\mathsf TH_{RT}
+B^\mathsf TH_{RR}B]Y.
$$

Equivalently, external columns have raw reference coefficients
$C_R^{\rm ext}=-BY$ and raw candidate coefficients $C_T^{\rm ext}=Y$.
The code contracts the six blocks with those transformations, avoiding a
physical full-space projector or an inverse overlap matrix.

Unprojected Gram groups are electronic determinant addresses; projected
groups are electronic translation orbits. Each group's Gram eigensolve is
independent, but reference/external thresholds are scaled by the complete
spectra across all groups. This preserves the global cutoff rule. The joint
reference/candidate Gram matrices must be positive semidefinite within the
independent roundoff tolerance. Group labels cannot justify discarding
Hamiltonian couplings between groups.

## 8. Block-Hamiltonian PT2 and response coefficients

Within the retained $\bar R\oplus Z$ space, choose

$$
H(\xi)=H_0+\xi V,
\qquad H_0=\begin{pmatrix}\bar H_R&0\\0&K\end{pmatrix},
\qquad V=\begin{pmatrix}0&W_{RQ}\\W_{RQ}^\mathsf T&0\end{pmatrix}.
$$

The external block retains its full interacting Hamiltonian. Diagonalizing
$\bar H_R$ gives the isolated lowest reference root
$\bar H_Ru_0=E_0u_0$ and $\Psi_0=\bar Ru_0$.
Since $V$ is off diagonal, $E^{(1)}=0$.
Expand $\Psi=\Psi_0+\xi\Psi_1+\cdots$ with intermediate normalization.
The first-order equation, projected onto $Z$, is

$$
v=W_{RQ}^\mathsf Tu_0,\qquad
A=K-E_0I,\qquad Aa=-v,\qquad \Psi_1=Za.
$$

The second-order energy is

$$
E^{(2)}=\langle\Psi_0|V|\Psi_1\rangle=v^\mathsf Ta,
\qquad E_{\rm MRPT2}=E_0+E^{(2)}.
$$

If $KQ=Q\operatorname{diag}(\kappa_\mu)$, the equivalent spectral formula is

$$
E^{(2)}=-\sum_\mu\frac{|(Q^\mathsf Tv)_\mu|^2}{\kappa_\mu-E_0},
\quad a=-Q\operatorname{diag}[(\kappa_\mu-E_0)^{-1}]Q^\mathsf Tv.
$$

The implementation obtains the spectrum for diagnostics and solves $Aa=-v$
directly; it never constructs $A^{-1}$. The raw physical coefficients are

$$
C^{(0)}_R=Xu_0,\qquad
C^{(1)}_R=C_R^{\rm ext}a,\qquad
C^{(1)}_T=C_T^{\rm ext}a.
$$

The first-order state generally has raw reference coefficients because
external candidates had reference components removed. It remains physically
orthogonal to the entire retained reference space. Its norm is $a^\mathsf Ta$.
Projected raw coefficients refer to seed columns; reconstructing an explicit
orbit state additionally applies Section 5's sign and character expansion.

The reference gap must exceed
$\epsilon_{\rm gap}\max(1,\max_j(e_j-E_0))$ and every external denominator
$\Delta_\mu=\kappa_\mu-E_0$ must exceed
$\epsilon_{\rm den}\max(1,\max_\mu|\Delta_\mu|)$.
Defaults are $10^{-10}$. These checks use energy differences and are
invariant to a common energy-origin shift, within numerical precision.
A one-dimensional reference has no excited-root gap check; empty $Q$ gives
zero correction. Failed gaps propagate as errors; no level shift,
pseudoinverse or per-point threshold relaxation is applied.

Positive denominators imply $E^{(2)}\leq0$, but $E_0+E^{(2)}$ is not a
variational upper bound. It can lie below converged ED. A small response
residual verifies the linear solve, not convergence of the perturbation
series. Near-degeneracy or a large first-order norm can limit the usefulness
of second order even with an accurately solved equation.

## 9. Numerical design, precision and cost

The data flow is:

1. Optimize or supply LF frame seeds, with a fixed physical model.
2. Enumerate site determinants and nonvacuum relative phonon configurations.
3. Assemble analytic $S/H$ blocks, optionally through relative translations.
4. Perform grouped Gram orthogonalization and reference subtraction.
5. Transform retained Hamiltonian blocks and solve the reference ground root.
6. Check spectra, solve the PT2 response and recover raw coefficients.
7. Independently check residuals/metrics; convergence scans occur outside the kernel.

`float64` preserves the original arithmetic. The optional
`reconditioned_longdouble` mode keeps the originally selected overlap spectra
and ranks and repairs coordinates **inside those retained spaces**. It
accumulates Gram/Hamiltonian products in `longdouble`, orthonormalizes the
retained reference coordinates, subtracts remaining reference leakage from
external coordinates, and Cholesky-normalizes their physical Gram matrices.
The raw kernels, stored coordinates and solver blocks remain `float64`.
It does not regenerate more accurate raw displacement kernels. Platforms
without extra `longdouble` precision raise an error for this mode; there is
no automatic fallback.

Physical metrics are evaluated against the full six-block raw Gram:
$\bar R^\mathsf T\bar R$, $Z^\mathsf TZ$ and $\bar R^\mathsf TZ$.
The SI object requires reference/external identity errors below $10^{-6}$,
cross-metric error below $10^{-7}$ and reference/response residuals below
$10^{-10}$. Roundoff symmetrization of transformed Hamiltonian blocks is
bounded and recorded; it does not change retained ranks or shift denominators.

Dense storage is $O(n_R^2+n_Rn_T+n_T^2)$. Grouped Gram eigensolves reduce
spectral work and temporary storage, but the raw candidate/candidate blocks
and final external response remain dense. For $L=4$, two seeds,
$N_\alpha=N_\beta=2$, $M=5$: $n_R=72$, $N=125$, $n_T=9000$.
One TT array is approximately 618 MiB and the pair of raw TT Gram/Hamiltonian
arrays is approximately 1.21 GiB before transformations and eigensolver
workspace. The historical full M5 batch reached about 3 GiB per worker.
A one-electron calculation has much smaller electronic dimension.
The implementation targets small clusters; arbitrary larger systems are not
made tractable by the object interface.

Cutoff evidence has separate meanings: ED solver convergence, ED oscillator
convergence, MP/MRPT phonon convergence, overlap-cut stability and
perturbative suitability. Historical MRPT2 phonon scans require two
successive energy increments below $10^{-6}$, numerical validity and
independent reference/external cutoff checks before setting their final
convergence flag. The new fixed-cutoff objects set `cutoff_converged=False`.
The bundled multi-electron M5 table passed its recorded numerical acceptance
but has no established phonon convergence; its rows remain labeled accordingly.

## 10. Density observables and symmetry analysis

For a site-diagonal electronic observable with value $O(I)$,

$$
\langle O\rangle=
\frac{\sum_I O(I)\sum_A C_{AI}(SC)_{AI}}{C^\mathsf TSC}.
$$

Consequently
$n_{p\sigma}=\sum_I n_{p\sigma}(I)w_I$ with
$w_I=\sum_A C_{AI}(SC)_{AI}/(C^\mathsf TSC)$.
Squaring individual nonorthogonal coefficients would omit cross-frame terms.
For a projected state, the density routine uses its expanded orbit
coefficients and the corresponding Gram matrix. The checks
$\sum_pn_{p\sigma}=N_\sigma$ and physical normalization apply to all sectors.

Examples report charge imbalance $\max_p n_p-\min_p n_p$ and spin amplitude
$\max_p|n_{p\alpha}-n_{p\beta}|$. In a one-electron spin-polarized sector,
a nonzero spin amplitude alone is not evidence of broken lattice symmetry;
the single-electron plot uses charge imbalance. Densities of a selected root
inside a nearly degenerate manifold are state-dependent and are flagged.
No MP2-corrected density is inferred by relabeling an HF density.

## 11. CS-HF and CS-MP2 extension

With an unrestricted determinant and a common coherent displacement $z$,
let $D^\sigma=C^\sigma_{\rm occ}(C^\sigma_{\rm occ})^\mathsf T$,
$n_{p\sigma}=D^\sigma_{pp}$ and $\rho_p=\sum_\sigma n_{p\sigma}$.
The energy before displacement elimination is

$$
E_{CS}=\sum_\sigma\operatorname{Tr}(tD^\sigma)
+U\sum_pn_{p\alpha}n_{p\beta}+\omega z^\mathsf Tz+2gz^\mathsf T\rho.
$$

Differentiation gives $z=-g\rho/\omega$ and

$$
E_{CS}=\sum_\sigma\operatorname{Tr}(tD^\sigma)
+U\sum_pn_{p\alpha}n_{p\beta}-\frac{g^2}{\omega}\rho^\mathsf T\rho,
\quad F^\sigma=t+\operatorname{diag}(Un_{\bar\sigma}+2gz).
$$

Orbital variations are $C^\sigma=C_0^\sigma e^{K^\sigma}$,
$K^\sigma_{ai}=\kappa^\sigma_{ai}$,
$K^\sigma_{ia}=-\kappa^\sigma_{ai}$. BFGS uses three-point derivatives.
The physical stationarity check is
$\sqrt{\sum_\sigma\|[F^\sigma,D^\sigma]\|_F^2}$.
Within occupied/virtual blocks separately, Fock diagonalization rotates
orbitals without changing the occupied projectors. BFGS termination and
physical convergence are reported separately; finite-difference precision
loss does not certify or preclude a stationary reference on its own.
The original one-electron density fixed-point driver remains available.

Choose the canonical Fock plus oscillator number operator as the CS
zeroth-order partition. A stationary coherent shift cancels the
one-phonon reference amplitude $\omega z_x+g\rho_x$.
Brillouin stationarity cancels zero-phonon electronic singles.
The mixed and opposite-spin double amplitudes are

$$
G^\sigma_{xia}=gC^\sigma_{xi}C^\sigma_{xa},\qquad
V_{iajb}=U\sum_p C^\alpha_{pi}C^\alpha_{pa}
C^\beta_{pj}C^\beta_{pb}.
$$

Thus

$$
E_{CS}^{(2)}=-\sum_{\sigma xia}
\frac{|G^\sigma_{xia}|^2}{\epsilon^\sigma_a-\epsilon^\sigma_i+\omega}
-\sum_{iajb}\frac{|V_{iajb}|^2}
{\epsilon^\alpha_a+\epsilon^\beta_b-\epsilon^\alpha_i-\epsilon^\beta_j}.
$$

Each opposite-spin double is counted once. On-site Hubbard interactions
have no same-spin double channel. At $g=0$ the expression reduces to
Hubbard unrestricted MP2; for one electron the double channel vanishes.
At $\lambda=0$ it agrees with the general LF formula when the phonon scalar
$\omega z_x$ is counted once. The SI LF-MP2 extension separates this scalar
from the electronic one-body matrix: tracing a one-electron representation
$\omega z_x I$ over $N_e$ occupied orbitals would incorrectly multiply it
by $N_e$. Existing zero-shift LF scans are unchanged by this correction.

## 12. Mapping equations to code

| Mathematical operation | Module / callable |
|---|---|
| Electronic strings, signs, tensor shape | `basis.make_electron_basis`, `basis.make_shape` |
| Configuration enumeration and displacement recurrence | `phonon.lf_total_phonon_configurations`, `phonon.displacement_number_matrix` |
| Displaced operator elements | `phonon.displaced_phonon_operator_matrices`, `phonon.multimode_phonon_operator_matrices` |
| Vacuum MR kernels and lowest NOCI root | `mr.noci.lf_multi_frame_overlap`, `lf_multi_frame_hamiltonian`, `lf_noci_lowest` |
| Direct real-character projection | `mr.noci.lf_translation_projected_noci` |
| Six raw MR/PT blocks | `mr.kernels.build_mr_candidate_blocks`, `build_projected_mr_candidate_blocks` |
| Reference subtraction and global cutoffs | `mr.space.orthogonalize_mr_spaces_blockwise` |
| Congruence transformations | `mr.space.transform_mr_hamiltonian_blocks` |
| Retained-space precision repair | `mr.precision.recondition_retained_spaces` |
| Spectral checks and response solve | `mr.pt2.solve_mrpt2_blocks` |
| Assembly-to-solver interface | `mr.pt2.mrpt2_from_blocks` |
| Density in the nonorthogonal metric | `observables.noci_site_density` |
| Multi-electron CS reference and channels | `scf.cs.cs_hf_multi_optimize`, `mp.cs.cs_mp2_unrestricted_reference_point` |
| Active electronic CI rotation and molecular basis actions | `point_group.apply_electronic_symmetry`, `molecular_point_group` |
| Dipole-derived Pauli-Fierz integrals with DSE | `molecular.MolecularPF` |

See [API.md](API.md) for constructor arguments and result fields, and the
[reference-data guide](../reference_data/README.md) for the numerical
provenance and the exact frame recipes used in the bundled figures.

## 13. Molecular electronic symmetry and the cavity-integral extension

The implemented molecular extension has two independently tested parts:
active electronic CI rotations and construction of Pauli-Fierz integral
data. The full derivation, parameter choices, software responsibilities and
runnable usage are in [Molecular symmetry and cavity guide](molecular_symmetry.md).
Section 14 builds on these utilities to implement joint electron/photon
symmetry projection and molecular LFMR reference selection.

### Active rotations and determinant signs

The active action is $R\phi_p=\sum_q\phi_qU_{qp}$. A nonorthogonal AO
action preserves the AO metric, with orthonormal representation

$$
U_{AO}^T S U_{AO}=S,\qquad
U_{orth}=(X^T S)U_{AO}X,\qquad X=S^{-1/2}.
$$

For ascending occupied-orbital lists within each spin sector,

$$
M_\sigma(U)[J,I]=\det U[J_{occ},I_{occ}],\qquad
C'=M_\alpha(U)C M_\beta(U)^T.
$$

This includes fermionic reordering signs. PySCF's passive CI transformation
therefore receives `U.T`; neither the complex CI coefficients nor their
normalization are changed otherwise. MolSym supplies finite-group operations
and irrep matrices, including complex characters and multidimensional irreps.
PySCF supplies spherical-AO and CI rotations. Linear molecules require an
explicit finite subgroup, such as D2h for H2.

### Total dipole, nuclear constant, and DSE

In the paper's electron-position-minus-nuclear-position convention, define

$$
\hat D_k=\sum_{pq\sigma}r^k_{pq}a^\dagger_{p\sigma}a_{q\sigma}
-d^{nuc}_k\hat 1,\qquad d^{nuc}_k=\sum_A Z_A(R_{A,k}-o_k).
$$

At fixed $N_e>0$, the one-body matrix
$\mu^k=r^k-d^{nuc}_k I/N_e$ represents this total operator exactly in that
sector. The division by $N_e$ subtracts the nuclear constant once. The model
must be rebuilt if its electron number changes. The default origin is the
nuclear-charge center; explicit origins are supplied in Bohr.

For a unit polarization $e_x$, $\mu_x=e_x\cdot\boldsymbol\mu$ and

$$
g^x=-\sqrt{\omega_x/2}\lambda_x\mu_x,\qquad
\Delta h=\frac12\sum_x\lambda_x^2\mu_x\mu_x,\qquad
\Delta V_{pqrs}=\sum_x\lambda_x^2\mu^x_{pq}\mu^x_{rs}.
$$

Normal ordering $\frac12\lambda_x^2\hat D_x^2$ produces the one-body
matrix product and the two-body term with the Hamiltonian's existing 1/2
prefactor. Products are formed in the fixed orthonormal basis. This is the
projected-dipole-product DSE of Cui Eqs. (7)-(9); the finite-basis quadrupole
correction in Eq. (10) is excluded. Nuclear dipole contributions are already
encoded in $\mu$, so `ecore` remains nuclear repulsion alone. There is no
photon zero-point constant and no lattice density-centering correction.

The first cavity example uses H2/STO-3G, R=0.74 Angstrom, a z-polarized mode
with $\omega=0.466751$ Ha and $\lambda_c=0.05$ or 0.5 au. Symmetric Lowdin
retains all AOs; it differs from the paper's meta-Lowdin choice. No additional
CCSD(T)-minus-MP2 electronic correction from the paper's Eq. (42) is added.
The example reports one saved LF-HF initialization, independent ED residuals,
and separate ED/MP2 photon-cutoff histories. Its convergence flags do not
prove a global HF minimum or a general LFMR reference-selection result.

## 14. Joint molecular cavity projection and fixed-pool LFMR

This extension supersedes the earlier implementation boundary stated in
Section 13. Joint electronic/photon representations, coherent LFMR
projection, optional analytic symmetry diagnostics and fixed-pool reference
selection are now implemented. The extended derivation and usage are in
[the molecular guide](molecular_symmetry.md). Install the standalone SI
package with `python -m pip install ".[symmetry]"` to obtain MolSym.

### A symmetry acts on orbitals and photon modes

Let $U_g$ be the active orthonormal-orbital action and $Q_g$ the active
photon-mode action, so a coherent amplitude transforms as
$\eta\mapsto Q_g\eta$. Orthogonal group representations preserve the
Hamiltonian only if

$$
U_g h U_g^T=h,\quad U_g^{\otimes4}V=V,\quad
Q_g\Omega=\Omega Q_g,\quad
U_g g^xU_g^T=\sum_yQ_g[y,x]g^y,
\qquad \Omega=\operatorname{diag}(\omega).
$$

`joint_symmetry` validates supplied representations against integral data.
For `MolecularPF`, `molecular_cavity_symmetry` starts with the bare
molecular finite group and filters to operations preserving its cavity.
Within each frequency block, use columns $A_x=\lambda_x e_x$. If
$O_gAA^TO_g^T=AA^T$, then

$$
Q_g=A^+O_gA+I-A^+A.
$$

The final term gives identity on uncoupled mode combinations. This admits
mixing of degenerate modes, rather than requiring each polarization to be
separately fixed. Actual integral covariance is checked after this
geometric criterion. The surviving operations must be closed; MolSym
supplies the subgroup label, canonical order and irreps. A charged
system with a custom dipole origin can require photon translations,
which are outside the current orthogonal action and can reduce the
accepted subgroup. Neutral common-origin shifts cancel in the total dipole.

### Relative kernels avoid rotating diagonal LF parameters

In the fixed Lowdin basis, a frame contains conditional displacements

$$
|A,I\rangle=|I\rangle|\eta_{AI}\rangle,\qquad
\eta_{AI,x}=z_{A,x}-\sum_p\lambda_{A,xp}n_p(I).
$$

An orbital mixture need not produce another density-diagonal LF frame.
Instead evaluate $K^{S/H}_{AI,BJ}(g)=\langle AI|1/H\,R(g)|BJ\rangle$.
With $a=\eta_{AI}$, $b=Q_g\eta_{BJ}$ and determinant action $T_g$,

$$
K^S(g)=e^{-\|a-b\|^2/2}(T_g)_{IJ},
$$

$$
K^H(g)=e^{-\|a-b\|^2/2}\left[
(H_{el}T_g)_{IJ}+\sum_x(a_x+b_x)(G_xT_g)_{IJ}
+\left(E_{core}+\sum_x\omega_xa_xb_x\right)(T_g)_{IJ}\right].
$$

$H_{el}$ contains h and V but excludes $E_{core}$. These formulas follow
from coherent overlap, linear-coordinate and number-operator matrix
elements; they use the full oscillator space. Relative kernels are
streamed one operation at a time and are not individually Hermitian.

For a d-dimensional irrep, use matrix-unit projectors

$$
P^\Gamma_{mn}=\frac{d}{|G|}\sum_gD^\Gamma_{mn}(g)^*R(g),\qquad
P_{mn}P_{kl}=\delta_{nk}P_{ml},\quad P_{mn}^\dagger=P_{nm}.
$$

Fix a partner row m and retain all columns n. The projected basis is
$P_{mn}|AI\rangle$, and its two-sided Hamiltonian matrix is

$$
H_{AnI,BkJ}=\langle AI|P_{nm}H P_{mk}|BJ\rangle
=\frac{d}{|G|}\sum_gD^\Gamma_{nk}(g)^*K^H_{AI,BJ}(g).
$$

The overlap has the identical reduction. H/S are independent of m; the
reconstructed partner state depends on m. `LFMR(model, lam, shift,
symmetry=group, irrep=label, row=0, determinants=None)` performs the
complex Hermitian NOCI solve with full coefficient shape
`(F,d,nstra,nstrb)`. Selected determinants instead give `(F,d,D)` and
must form a subspace closed under electronic group actions. Selected
rows contain alpha/beta bit integers; full CI axes contain determinant
addresses. Exact MolSym labels are required, e.g. `A_1g` for D2h and
`A_1'` for D3h. The real lattice `character` path remains available and
cannot be combined with molecular symmetry.

The current dense guards are $D\le256$ and $FdD\le1024$. Flattening is
confined to the solve boundary and conjugate transposes replace real
transposes. Null metric directions are removed before the eigenproblem;
the reported projected residual applies to that retained variational
space, not the full physical Hilbert space. Frames are fixed: there is
no variation after projection, spin projection or molecular MRPT2.

### Partner transformation, purity and reference choice

For a normalized fixed-row state, the actual group expectation is
$r_g=\langle\Psi_m|R(g)|\Psi_m\rangle=D^\Gamma_{mm}(g)$. Thus

$$
p_\Gamma=\frac{d}{|G|}\sum_g\chi_\Gamma(g)^*r_g=1.
$$

The optional `symmetry_diagnostics()` contracts the state with analytic
relative overlap kernels to measure these expectations, central
representation purity and norm consistency. Only the totally symmetric
one-dimensional irrep is individually invariant under every operation.
Non-Abelian partner rows generally mix under the group.

`select_lf_references` compares a fixed pool with `unprojected`,
`projected` and `greedy` strategies. The first two rank single-frame CI
energies before or after projection; the third evaluates actual MR
lowering. Every accepted sequence uses the same target sector. Candidate
columns C are residualized against retained columns B in the physical
Gram metric,

$$
C_\perp=C-B(B^\dagger S B)^{-1}B^\dagger SC.
$$

Only new residual directions are thresholded, using the fixed scale of
their original candidate block. Old directions are retained unchanged,
so accepted spaces are nested. Duplicate/zero projected spaces add no
direction. The default stopping gain is $10^{-8}$ Ha after the first
accepted frame. This is fixed-pool selection, not reference optimization;
the entire pool must fit the dense guards before selection begins.

Initial method-test parameters are H2/STO-3G at R=0.74, 1.5 and 2.0
Angstrom with one bond-polarized mode, $\omega=0.466751$ Ha; and
equilateral H3+/STO-3G at side lengths 0.9 and 1.5 Angstrom with two
equal in-plane modes, $\omega=0.5$ Ha. Cavity strengths are
$\lambda_c=0,0.05,0.5$ au. The runnable `benchmarks/si/molecular_lfmr.py` in the research repository
CLI exposes `--molecule h2/h3`, `--distance`, `--lambda-c`,
`--output`, `--ed-cutoffs`, and `--suite` for all 15 combinations. The
[molecular guide](molecular_symmetry.md#10-initial-calculations-and-independent-acceptance)
specifies the three HF initial guesses, trial perturbations, solver settings
and saved files. Coherent LFMR and photon-truncated ED require separate
error checks. A total-photon-number cutoff preserves degenerate-mode
rotation symmetry, whereas a rectangular per-mode cutoff generally does
not. ED residuals, cutoff convergence, metric stability and irrep purity
must be recorded separately.

For nearly dependent retained columns, reduced H/S products use extended
accumulation when column magnitudes exceed 100, then LAPACK solves the
actual reduced generalized eigenproblem in complex128. Retained residuals
and norms use extended accumulation on the saved coefficient vector.
This does not regenerate more accurate raw kernels or alter rank cutoffs.

The first 15-point study has passed the recorded numerical checks. Its
parameters, candidate provenance and representative strategy comparisons
are documented in [the molecular guide](molecular_symmetry.md#recorded-first-study).
This validates the implementation at those small-basis points; the
remaining strong-coupling H3+ variational error is not a solver residual.

## 15. Translation and spin-exchange reduction of lattice MRPT2

The merged `mr_lf.mr.reduced` implements an alternative storage path for
the lattice block-Hamiltonian partition. It does not implement molecular
MRPT2. External configurations must be unique int64 rows and closed under
lattice translation; supplying a nonclosed list is rejected.

Let B contain the complete projected external candidates. If a seed is
exactly translation invariant, several columns differ only by a sign.
Replacing each orbit of k equivalent columns by one representative weighted
by sqrt(k) preserves the candidate operator:

$$
B B^\dagger=B_{red}B_{red}^\dagger.
$$

Thus this removal of exact redundancy preserves all nonzero Gram
eigenvalues as well as the represented space. A finite external eigenvalue
cutoff therefore keeps the same threshold and physical directions.

For equal spin counts the exchange operator obeys

$$
F|I_\alpha,I_\beta\rangle
=(-1)^{N_\alpha N_\beta}|I_\beta,I_\alpha\rangle,
\qquad F^2=1.
$$

Density-dependent LF frames and the lattice Hamiltonian commute with F.
Only the external parity sector matching the reference root couples to
that root. When removing the other sector, the cutoff still uses the
largest reference-subtracted Gram eigenvalue from both sectors:

$$
\tau_T=\epsilon_T\max(1,s_{max}^{+},s_{max}^{-}).
$$

`external_gram_maximum` supplies the omitted-sector scale. Reference
directions stay complete. Electronic-orbit groups give block-diagonal
overlap storage; Hamiltonian couplings between groups are all retained.
`defer_h=True` builds each Hamiltonian group pair only when it is contracted.
`lean_mrpt2` uses float64, checks reference gaps and positive denominators,
and returns residual and metric diagnostics. The external workflow applies
the same numerical acceptance thresholds as `LFMRPT2`.

Independent tests in the research repository compare raw reduced blocks
with complete-list congruences, and compare full/reduced/lean response
energies on small cutoffs. This integration does not certify a new full
high-cutoff alpha scan or its convergence. The resumable convergence and
plot drivers live under `benchmarks/si/` in the research repository.
