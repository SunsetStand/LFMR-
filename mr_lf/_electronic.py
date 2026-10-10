"""PySCF electronic contractions shared by the general integral paths."""

import numpy as np
from pyscf.fci import addons, cistring, direct_nosym


class ElectronicSpace:
    def __init__(self, model):
        self.model = model
        n = model.norb
        if n >= 64:
            raise NotImplementedError("the determinant representation requires norb < 64")
        for permutation in ((2, 3, 0, 1), (1, 0, 3, 2)):
            if not np.allclose(model.eri, model.eri.transpose(permutation), atol=1e-12, rtol=0):
                raise ValueError("eri must have pair-exchange and real Hermitian symmetry")
        self.strings = [cistring.make_strings(range(n), count) for count in model.nelec]
        self.links = tuple(cistring.gen_linkstr_index(range(n), count) for count in model.nelec)
        self.shape = tuple(len(s) for s in self.strings)
        self.size = int(np.prod(self.shape))
        self.occupations = [((s[:, None] >> np.arange(n)) & 1).astype(float)
                            for s in self.strings]
        self.density = (self.occupations[0][:, None] + self.occupations[1][None, :]).reshape(-1, n)
        self.h2 = (direct_nosym.absorb_h1e(model.h1e, model.eri, n, model.nelec, .5)
                   if sum(model.nelec) else None)

    def apply(self, ci):
        if self.h2 is None:
            return np.zeros(self.shape)
        return direct_nosym.contract_2e(
            self.h2, np.ascontiguousarray(ci), self.model.norb, self.model.nelec, self.links
        )

    def apply_one(self, operator, ci):
        return direct_nosym.contract_1e(
            np.ascontiguousarray(operator), np.ascontiguousarray(ci),
            self.model.norb, self.model.nelec, self.links,
        )

    def diagonal(self):
        """Independent Slater–Condon diagonal, excluding the scalar offset.

        sum h_pp n_p + 1/2 sum (pp|qq)n_p n_q
        - 1/2 sum_sigma (pq|qp)n_p,sigma n_q,sigma.
        Linear coupling has zero number-state diagonal.
        """
        m = self.model
        p = np.arange(m.norb)
        coulomb = m.eri[p[:, None], p[:, None], p[None, :], p[None, :]]
        exchange = m.eri[p[:, None], p[None, :], p[None, :], p[:, None]]
        diagonal = self.density @ np.diag(m.h1e)
        diagonal += .5 * np.einsum("ip,pq,iq->i", self.density, coulomb, self.density)
        for spin, occ in enumerate(self.occupations):
            term = .5 * np.einsum("ip,pq,iq->i", occ, exchange, occ)
            diagonal -= np.broadcast_to(term[:, None] if spin == 0 else term[None, :],
                                        self.shape).ravel()
        return diagonal.reshape(self.shape)

    def dense(self, *, limit=256):
        """Small electronic oracle for finite-space LF-MP2, never used by ED."""
        if self.size > limit:
            raise NotImplementedError(f"finite-space LF-MP2 requires <= {limit} electronic determinants")
        identity = np.eye(self.size)
        ham = np.column_stack([self.apply(c.reshape(self.shape)).ravel() for c in identity])
        coupling = np.array([
            np.column_stack([self.apply_one(g, c.reshape(self.shape)).ravel() for c in identity])
            for g in self.model.coupling
        ]).reshape(self.model.nmode, self.size, self.size)
        return ham, coupling

    def reference_ci(self, mo_coeff):
        reference = np.zeros(self.shape)
        reference[0, 0] = 1
        return addons.transform_ci_for_orbital_rotation(
            reference, self.model.norb, self.model.nelec, [c.T for c in mo_coeff]
        )

