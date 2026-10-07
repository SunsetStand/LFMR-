"""Site determinant enumeration and C-order tensor indexing."""

import numpy as np
from pyscf.fci import cistring


def encode(occupations, d):
    if d <= 0 or isinstance(d, (int, np.integer)) == False:
        raise ValueError("d must be positive and an integer")
    for i in occupations:
        if i < 0 or i >= d or isinstance(i, (int, np.integer)) == False:
            raise ValueError("occupation must be less than d, non-negative and an integer")
    p = 0
    for i in occupations:
        p = p * d + i
    return p


def decode(p, L, d):
    if d <= 0 or isinstance(d, (int, np.integer)) == False:
        raise ValueError("d must be positive and an integer")
    if L <= 0 or isinstance(L, (int, np.integer)) == False:
        raise ValueError("L must be positive and an integer")
    if p < 0 or p >= d**L or isinstance(p, (int, np.integer)) == False:
        raise ValueError("p must be less than d**L, non-negative and an integer")
    occupations = ()
    for i in range(L):
        occupations = (p % d,) + occupations
        p = p // d
    return occupations


def make_electron_basis(L, N):
    if N < 0 or N > L or isinstance(N, (int, np.integer)) == False:
        raise ValueError("N must be non-negative, no greater than L and an integer")
    if L <= 0 or L >= 64 or isinstance(L, (int, np.integer)) == False:
        raise ValueError("L must be positive, less than 64 and an integer")
    strings = cistring.make_strings(range(L), N)
    links = cistring.gen_linkstr_index(range(L), N, strings)
    return strings, links


def make_shape(L, nelec, Nmax):
    if L <= 0 or L >= 64 or isinstance(L, (int, np.integer)) == False:
        raise ValueError("L must be positive, less than 64 and an integer")
    if isinstance(nelec, tuple) == False or len(nelec) != 2:
        raise ValueError("nelec must be a tuple of 2 elements")
    for n in nelec:
        if n < 0 or n > L or isinstance(n, (int, np.integer)) == False:
            raise ValueError("nelec must be a tuple of integer between 0 and L")
    if Nmax < 0 or isinstance(Nmax, (int, np.integer)) == False:
        raise ValueError("Nmax must be non-negative and an integer")
    neleca, nelecb = nelec
    num_strings_a = cistring.num_strings(L, neleca)
    num_strings_b = cistring.num_strings(L, nelecb)
    d = Nmax + 1
    return (num_strings_a, num_strings_b) + (d,) * L
