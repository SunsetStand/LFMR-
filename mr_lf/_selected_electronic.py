"""Electronic contractions on an ordered, arbitrary list of determinants.

The list contains paired alpha/beta bit strings, not FCI addresses or a
Cartesian product of spin-string lists.  Only final bra states are filtered:
the intermediate state in E_pq E_rs may lie outside the selected space.
No complete fixed-electron-number space is enumerated or allocated.
"""

import numpy as np
from pyscf.fci import cistring

from .model import ElectronPhononModel


class SelectedElectronicSpace:
    """Project electronic operators onto at most 256 paired determinants.

    ``determinants`` is a strict int64 array of shape (D,2), ordered alpha
    bit string, beta bit string.  All contractions use finite float64
    vectors of shape (D,); the electronic Hamiltonian excludes ``ecore``.
    Real chemists' ERIs retain all four indices and require only the same
    pair-exchange and real Hermitian symmetry as ``ElectronicSpace``.
    """

    determinant_limit = 256

    def __init__(self, model, determinants):
        if not isinstance(model, ElectronPhononModel):
            raise TypeError("model must be an ElectronPhononModel or MolecularEP")
        if model.norb >= 64:
            raise NotImplementedError("the determinant representation requires norb < 64")
        if not isinstance(determinants, np.ndarray) or determinants.dtype != np.int64:
            raise TypeError("determinants must be an int64 NumPy array")
        if determinants.ndim != 2 or determinants.shape[1] != 2:
            raise ValueError("determinants must have shape (D,2)")
        size = len(determinants)
        if size < 1:
            raise ValueError("determinants must contain at least one paired determinant")
        if size > self.determinant_limit:
            raise NotImplementedError(
                f"selected CI requires <= {self.determinant_limit} electronic determinants")
        bound = 1 << model.norb
        lookup = {}
        for address, values in enumerate(determinants):
            pair = tuple(int(s) for s in values)
            if any(s < 0 or s >= bound for s in pair):
                raise ValueError("determinant bit strings must be nonnegative and within norb")
            if any(s.bit_count() != count for s, count in zip(pair, model.nelec)):
                raise ValueError("determinant bit strings must have the model's electron counts")
            if pair in lookup:
                raise ValueError("determinants must not contain duplicate alpha/beta pairs")
            lookup[pair] = address
        for permutation in ((2, 3, 0, 1), (1, 0, 3, 2)):
            if not np.allclose(model.eri, model.eri.transpose(permutation), atol=1e-12, rtol=0):
                raise ValueError("eri must have pair-exchange and real Hermitian symmetry")
        self.model = model
        self.size = size
        self.shape = (size,)
        self.determinants = np.array(determinants, dtype=np.int64, order="C", copy=True)
        self.determinants.setflags(write=False)
        self._lookup = lookup
        bits = (self.determinants[:, :, None] >> np.arange(model.norb)) & 1
        self.density = np.sum(bits, axis=1, dtype=np.float64)
        # a_p^dagger a_r^dagger a_s a_q = E_pq E_rs - delta_qr E_ps
        # after the spin sums.  The contraction uses the full ordered ERI;
        # no extra Coulomb symmetry or pair packing is introduced.
        with np.errstate(over="ignore", invalid="ignore"):
            self._one_electron = model.h1e - .5 * np.einsum("pqqs->ps", model.eri)
        if not np.all(np.isfinite(self._one_electron)):
            raise ValueError("effective one-electron integrals are nonfinite")

    @staticmethod
    def _float64_array(array, shape, name):
        if not isinstance(array, np.ndarray) or array.dtype != np.float64:
            raise TypeError(f"{name} must be a float64 NumPy array")
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    def _one_body_actions(self, alpha, beta):
        """Yield p,q,final-pair,sign for each allowed spin contribution."""
        n = self.model.norb
        for spin, string in enumerate((alpha, beta)):
            for q in range(n):
                if not string & (1 << q):
                    continue
                for p in range(n):
                    if p != q and string & (1 << p):
                        continue
                    # Occupation checks precede cre_des_sign: its p=q
                    # special case returns one even for an empty orbital.
                    sign = cistring.cre_des_sign(p, q, string)
                    target = string ^ (1 << p) ^ (1 << q)
                    pair = (target, beta) if spin == 0 else (alpha, target)
                    yield p, q, pair, sign

    def _one_body_contributions(self, pair, operator):
        for p, q, target, sign in self._one_body_actions(*pair):
            value = operator[p, q] * sign
            if value:
                row = self._lookup.get(target)
                if row is not None:
                    yield row, value

    def _hamiltonian_contributions(self, pair):
        yield from self._one_body_contributions(pair, self._one_electron)
        for r, s, intermediate, first_sign in self._one_body_actions(*pair):
            plane = self.model.eri[:, :, r, s]
            if not np.any(plane):
                continue
            # Do not check intermediate in _lookup.  P E_pq E_rs P is
            # generally different from P E_pq P E_rs P.
            for p, q, target, second_sign in self._one_body_actions(*intermediate):
                value = .5 * plane[p, q] * first_sign * second_sign
                if value:
                    row = self._lookup.get(target)
                    if row is not None:
                        yield row, value

    def _apply_contributions(self, ci, contributions):
        result = np.zeros(self.shape, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for ket, amplitude in enumerate(ci):
                if amplitude:
                    pair = tuple(int(s) for s in self.determinants[ket])
                    for row, value in contributions(pair):
                        result[row] += value * amplitude
        if not np.all(np.isfinite(result)):
            raise ValueError("electronic contraction produced nonfinite values")
        return result

    def apply(self, ci):
        """Return P H_el P ci without constructing a dense electronic matrix."""
        ci = self._float64_array(ci, self.shape, "ci")
        return self._apply_contributions(ci, self._hamiltonian_contributions)

    def apply_one(self, operator, ci):
        """Contract a spin-summed one-body operator, symmetric or otherwise."""
        operator = self._float64_array(
            operator, (self.model.norb, self.model.norb), "operator")
        ci = self._float64_array(ci, self.shape, "ci")
        return self._apply_contributions(
            ci, lambda pair: self._one_body_contributions(pair, operator))

    def dense(self, limit=256):
        """Assemble selected electronic/coupling matrices from contributions.

        Returns float64 arrays (D,D) and (nmode,D,D), with no scalar
        offset.  This small-space method never calls complete-FCI routines
        or constructs a complete-FCI vector, and ``apply`` does not use it.
        """
        if (isinstance(limit, (bool, np.bool_))
                or not isinstance(limit, (int, np.integer)) or limit < 1):
            raise ValueError("limit must be a positive integer")
        if self.size > limit:
            raise NotImplementedError(
                f"selected electronic matrices require <= {limit} determinants")
        ham = np.zeros((self.size, self.size), dtype=np.float64)
        coupling = np.zeros((self.model.nmode, self.size, self.size), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for ket, values in enumerate(self.determinants):
                pair = tuple(int(s) for s in values)
                for row, value in self._hamiltonian_contributions(pair):
                    ham[row, ket] += value
                for x, operator in enumerate(self.model.coupling):
                    for row, value in self._one_body_contributions(pair, operator):
                        coupling[x, row, ket] += value
        if not np.all(np.isfinite(ham)) or not np.all(np.isfinite(coupling)):
            raise ValueError("selected electronic matrices contain nonfinite values")
        return ham, coupling
