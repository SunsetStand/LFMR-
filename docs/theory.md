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

`float64` is used for physical kernels and state coefficients; configuration
lists and group labels use `int64`. Array orders are:

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

The general integral paths now support ED and a limited LF-HF/MP2
calculation. The lattice formulas in the remainder of this document keep
their original specialization. General MR/PT2, molecular symmetry restoration,
Hessians and reference-selection studies are subsequent work.

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

See [API.md](API.md) for constructor arguments and result fields, and the
[reference-data guide](../reference_data/README.md) for the numerical
provenance and the exact frame recipes used in the bundled figures.
