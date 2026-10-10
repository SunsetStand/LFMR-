"""Hubbard–Holstein LF methods and molecular electronic–phonon integral adapters."""

from .exact import ExactED
from .model import ElectronPhononModel, HubbardHolstein
from .molecular import MolecularEP
from .mp.cs import CSMP2
from .mp.lf import LFMP2
from .mr.noci import LFMR
from .mr.pt2 import LFMRPT2
from .scf.cs import CSHF
from .scf.lf import LFHF

__version__ = "1.0.0"
__authors__ = "Chenxi Wang & Zhihao Cui (advisor)"
__all__ = [
    "CSHF",
    "CSMP2",
    "LFHF",
    "LFMP2",
    "LFMR",
    "LFMRPT2",
    "ElectronPhononModel",
    "ExactED",
    "HubbardHolstein",
    "MolecularEP",
]
