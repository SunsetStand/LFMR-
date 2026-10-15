"""Real electronic integrals and uncentered linear electron–phonon models."""

from dataclasses import dataclass

import numpy as np


def _real_array(values, name):
    """Own a finite, real numeric array without silently discarding complex parts."""
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    array = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _symmetric_array(array, name):
    transpose = array.swapaxes(-1, -2)
    if not np.allclose(array, transpose, atol=1e-12, rtol=0):
        raise ValueError(f"{name} must be symmetric in its last two indices")
    return array * 0.5 + transpose * 0.5


def _validate_phonons(omega, coupling, norb, *, coupling_name="coupling"):
    omega = _real_array(omega, "omega")
    if omega.ndim != 1 or np.any(omega <= 0):
        raise ValueError("omega must be a one-dimensional array of positive frequencies")
    coupling = _real_array(coupling, coupling_name)
    shape = (omega.size, norb, norb)
    if coupling.shape != shape:
        raise ValueError(f"{coupling_name} must have shape {shape}, [mode,orbital,orbital]")
    return omega, _symmetric_array(coupling, coupling_name)


@dataclass
class ElectronPhononModel:
    r"""Integral data in a fixed orthonormal electronic basis, in Hartree.

    The Hamiltonian is

        ecore + sum(h1e[p,q] a_p^dagger a_q)
        + 1/2 sum(eri[p,q,r,s] a_p^dagger a_r^dagger a_s a_q)
        + sum(omega[x] b_x^dagger b_x)
        + sum(coupling[x,p,q] a_p^dagger a_q (b_x + b_x^dagger)),

    with the corresponding spin sums on electronic operators. ``eri`` uses
    chemists' (pq|rs) index order and is kept as a full four-index tensor;
    no Coulomb permutation symmetry is imposed or used to pack it.
    ``coupling`` is uncentered, with no phonon zero-point constant or hidden
    frequency/mass factors. ``ecore`` is a separate scalar energy offset.

    h1e: (norb,norb); eri: (norb,norb,norb,norb);
    omega: (nmode,); coupling: (nmode,norb,norb).
    All arrays are owned C-order float64 copies. norb and nmode are
    independent; nmode=0 uses omega.shape=(0,) and coupling.shape=(0,norb,norb).
    nelec is an explicit (N_alpha,N_beta) tuple. ExactED, LFHF, LFMP2 and
    fixed-frame LFMR accept this integral interface.
    """

    h1e: np.ndarray
    eri: np.ndarray
    omega: np.ndarray
    coupling: np.ndarray
    nelec: tuple[int, int]
    ecore: float = 0.0

    def __post_init__(self):
        h1e = _real_array(self.h1e, "h1e")
        if h1e.ndim != 2 or h1e.shape[0] == 0 or h1e.shape[0] != h1e.shape[1]:
            raise ValueError("h1e must have nonempty square shape (norb,norb)")
        self.h1e = _symmetric_array(h1e, "h1e")
        self.eri = _real_array(self.eri, "eri")
        if self.eri.shape != (self.norb,) * 4:
            raise ValueError("eri must have full shape (norb,norb,norb,norb)")
        self.omega, self.coupling = _validate_phonons(self.omega, self.coupling, self.norb)
        if (
            not isinstance(self.nelec, tuple)
            or len(self.nelec) != 2
            or any(
                isinstance(n, (bool, np.bool_))
                or not isinstance(n, (int, np.integer))
                or not 0 <= n <= self.norb
                for n in self.nelec
            )
        ):
            raise ValueError("nelec must be (N_alpha,N_beta), each in [0,norb]")
        self.nelec = tuple(int(n) for n in self.nelec)
        if (
            isinstance(self.ecore, (bool, np.bool_))
            or not isinstance(self.ecore, (int, float, np.integer, np.floating))
            or not np.isfinite(self.ecore)
        ):
            raise ValueError("ecore must be a finite real scalar")
        self.ecore = float(self.ecore)

    @property
    def norb(self):
        return self.h1e.shape[0]

    @property
    def nmode(self):
        return self.omega.size


def alpha_to_g(alpha, omega):
    if alpha < 0 or omega <= 0:
        raise ValueError("alpha should be non-negative and omega should be positive")
    return (alpha * omega) ** (0.5)


def electron_ring_hopping(L, t=-1.0):
    if L < 3 or isinstance(L, (int, np.integer)) == False:
        raise ValueError("L must be an integer greater than 2")
    Ht = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            if i == (j - 1) % L or i == (j + 1) % L:
                Ht[i, j] = t
    return Ht


@dataclass
class HubbardHolstein:
    """One real phonon mode per site; no phonon zero-point energy.

    tmat[p,q] is the hopping from q to p; nelec=(N_alpha,N_beta).
    Array inputs are copied to float64. Only real parameters are supported.
    """

    tmat: np.ndarray
    U: float
    g: float
    omega: float
    nelec: tuple[int, int]

    def __post_init__(self):
        if np.iscomplexobj(self.tmat):
            raise TypeError("complex hopping is not supported")
        t = np.asarray(self.tmat, dtype=np.float64)
        if t.ndim != 2 or t.shape[0] != t.shape[1] or not 0 < len(t) < 64:
            raise ValueError("tmat must be square with 0 < L < 64")
        if not np.all(np.isfinite(t)) or not np.allclose(t, t.T, atol=1e-12, rtol=0):
            raise ValueError("tmat must be finite and symmetric")
        # The physical model owns an exactly symmetric matrix; tolerate only
        # roundoff in a caller's Hermitian construction.
        self.tmat = (t + t.T) * 0.5
        for name in ("U", "g", "omega"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and real")
            setattr(self, name, float(value))
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        if (
            not isinstance(self.nelec, tuple)
            or len(self.nelec) != 2
            or any(
                isinstance(n, (bool, np.bool_))
                or not isinstance(n, (int, np.integer))
                or not 0 <= n <= self.L
                for n in self.nelec
            )
        ):
            raise ValueError("nelec must be (N_alpha,N_beta), each in [0,L]")
        self.nelec = tuple(int(n) for n in self.nelec)

    @property
    def L(self):
        return len(self.tmat)

    @property
    def alpha(self):
        return self.g**2 / self.omega

    def to_general(self):
        """Copy this model into the full orthonormal integral representation.

        eri[p,p,p,p]=U gives U*n[p,alpha]*n[p,beta] with the Hamiltonian's
        factor 1/2 and spin sums. coupling[x,p,q]=g*delta[x,p]*delta[p,q].
        No centering correction or zero-point energy is added.
        """
        sites = np.arange(self.L)
        eri = np.zeros((self.L,) * 4, dtype=np.float64)
        eri[sites, sites, sites, sites] = self.U
        coupling = np.zeros((self.L,) * 3, dtype=np.float64)
        coupling[sites, sites, sites] = self.g
        return ElectronPhononModel(
            self.tmat, eri, np.full(self.L, self.omega), coupling, self.nelec
        )

    @classmethod
    def ring(cls, L, *, nelec=(1, 0), t=-1.0, U=0.0, omega=0.5, alpha=None, g=None):
        """Periodic L>=3 chain; specify either alpha or g (default zero)."""
        if alpha is not None and g is not None:
            raise ValueError("specify alpha or g, not both")
        if alpha is not None:
            if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha):
                raise ValueError("alpha must be finite and nonnegative")
            g = alpha_to_g(alpha, omega)
        return cls(electron_ring_hopping(L, t), U, 0.0 if g is None else g, omega, nelec)
