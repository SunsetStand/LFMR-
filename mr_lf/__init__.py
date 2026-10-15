"""Hubbard–Holstein LF methods and molecular electronic–phonon integral adapters."""

from .exact import ExactED
from .joint_symmetry import JointSymmetry, joint_symmetry, molecular_cavity_symmetry
from .model import ElectronPhononModel, HubbardHolstein
from .molecular import MolecularEP, MolecularPF
from .mp.cs import CSMP2
from .mp.lf import LFMP2
from .mr.noci import LFMR
from .mr.pt2 import LFMRPT2
from .mr.reference_selection import select_lf_references
from .point_group import apply_electronic_symmetry, molecular_point_group, point_group_data
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
    "JointSymmetry",
    "MolecularEP",
    "MolecularPF",
    "apply_electronic_symmetry",
    "joint_symmetry",
    "molecular_cavity_symmetry",
    "molecular_point_group",
    "point_group_data",
    "select_lf_references",
]
