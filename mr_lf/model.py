"""Physical Hubbard–Holstein model with uncentered density coupling."""

from dataclasses import dataclass

import numpy as np


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
