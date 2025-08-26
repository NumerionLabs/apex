# Standard
import logging
import re
from typing import Optional

# Third party
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, Mol

STICKY_DICT = {"[U]": "%50", "[Np]": "%51", "[Pu]": "%52", "[Am]": "%53"}
ELEMENT_LIST = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


def get_sticky_smiles(smiles: str) -> str:
    """
    Transforms a synthon SMILES with metallic attachment points into the
    original string-based "sticky" format with numerical attachment points,
    which enables faster, string-based construction of products.
    """
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), kekuleSmiles=True)
    for sticky_symbol in STICKY_DICT.keys():
        if smiles.startswith(sticky_symbol):
            # Create a regex pattern to find the first element symbol in the
            # SMILES
            element_pattern = "|".join(
                re.escape(symbol) for symbol in ELEMENT_LIST
            )
            match = re.search(element_pattern, smiles[len(sticky_symbol) :])

            if match:
                # Get the positions of the sticky symbol and the first
                # element symbol
                element_start = match.start() + len(sticky_symbol)
                element = match.group()

                # Construct the new SMILES by swapping the sticky symbol and
                # the element symbol
                smiles = (
                    element
                    + smiles[len(sticky_symbol) : element_start]
                    + sticky_symbol
                    + smiles[element_start + len(element) :]
                )
            continue

    # Replace symbols with placeholders and build suffix
    suffix_parts = []
    for symbol, replacement in STICKY_DICT.items():
        if symbol in smiles:
            suffix_parts.append(f"[{symbol[1:-1]}]{replacement}")
            smiles = smiles.replace(symbol, replacement)

            # Remove parentheses around replacements with bond modifiers
            smiles = re.sub(
                r"\(([-=#$]?)" + re.escape(replacement) + r"\)",
                r"\1" + replacement,
                smiles,
            )

    # Construct the suffix from replacements
    suffix = "." + ".".join(suffix_parts) if suffix_parts else ""

    return smiles + suffix


def _get_return(i, types) -> tuple[int, int]:
    return {i: j for j, i in enumerate(types)}[i], len(types)


def _one_hot_featurizer(idx: int, length: int) -> list[bool]:
    one_hot = [False] * length
    one_hot[idx] = True
    return one_hot


def get_default_allowed_elements() -> list[str]:
    return [
        "*",
        "B",
        "Br",
        "C",
        "Cl",
        "F",
        "Fe",
        "I",
        "N",
        "O",
        "P",
        "S",
        "Se",
        "Si",
        "Sn",
    ]


def get_default_sticky_elements() -> list[str]:
    return ["U", "Np", "Pu", "Am", "*"]


def reassign_sticky_elements(
    mol: Mol,
    inplace: bool = True,
    sticky_elements: Optional[list[str]] = None,
) -> Mol:
    """
    Reassigns a sticky element (i.e., an attachment point) on an RDKit molecule.
    In the reassignment, the sticky element atomic number gets updated to zero
    (hence, it will have a symbol of "*") and its charge is reset.

    Args:
        mol: The input RDKit molecule.
        inplace: A boolean that specifies whether the reassignment modifies the
          molecule in place or not.
        sticky_elements: An optional list of strings indicating the symbols that
          correspond to sticky elements.

    Returns:
        The RDKit molecule with attachment points reassigned to the "*"
    """
    sticky_elements = sticky_elements or get_default_sticky_elements()
    if not inplace:
        mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in sticky_elements:
            atom.SetAtomicNum(0)
            atom.SetFormalCharge(0)
    return mol


def prepare_molecule(mol: Mol, remove_hydrogens: bool = True) -> Mol:
    p = Chem.AdjustQueryParameters.NoAdjustments()
    p.makeDummiesQueries = True
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol)
    mol = reassign_sticky_elements(mol, False)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = Chem.AdjustQueryProperties(mol, p)
    return mol


def get_atom_element_idx(atom: Atom) -> tuple[int, int]:
    i = atom.GetSymbol()
    types = get_default_allowed_elements()
    if (i in get_default_sticky_elements()) or (i not in types):
        i = "*"
    return _get_return(i, types)


def get_atom_degree_idx(atom: Atom) -> tuple[int, int]:
    i = min(max(1, atom.GetDegree()), 7)
    types = list(range(1, 8))
    return _get_return(i, types)


def get_atom_hybridization_idx(atom: Atom) -> tuple[int, int]:
    i = atom.GetHybridization().name
    types = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]
    return _get_return(i, types)


def get_atom_chirality_idx(atom: Atom) -> tuple[int, int]:
    i = atom.GetChiralTag().name
    types = [
        "CHI_UNSPECIFIED",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
    ]
    return _get_return(i, types)


def get_atom_hydrogens_idx(atom: Atom) -> tuple[int, int]:
    i = min(atom.GetTotalNumHs(), 7)
    types = list(range(8))
    return _get_return(i, types)


def get_atom_formal_charge_idx(atom: Atom) -> tuple[int, int]:
    i = min(max(-3, atom.GetFormalCharge()), 3)
    types = list(range(-3, 4))
    return _get_return(i, types)


def get_atom_aromaticity_idx(atom: Atom) -> tuple[int, int]:
    i = atom.GetIsAromatic()
    types = [False, True]
    return _get_return(i, types)


def get_bond_order_idx(bond: Bond) -> tuple[int, int]:
    i = bond.GetBondType().name
    types = ["SINGLE", "DOUBLE", "TRIPLE", "UNSPECIFIED"]
    return _get_return(i, types)


def get_bond_conjugated_idx(bond: Bond) -> tuple[int, int]:
    i = bond.GetIsConjugated()
    types = [False, True]
    return _get_return(i, types)


def get_bond_ring_idx(bond: Bond) -> tuple[int, int]:
    i = bond.IsInRing()
    types = [False, True]
    return _get_return(i, types)


def get_bond_stereo_idx(bond: Bond) -> tuple[int, int]:
    i = bond.GetStereo().name
    types = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOTRANS"]
    return _get_return(i, types)


def get_atom_features(atom: Atom) -> list[bool]:
    assert Vocabulary.ATOM_VOCAB is not None
    out = []
    for func_name in Vocabulary.ATOM_VOCAB:
        idx, length = ATOM_FUNCTIONS[func_name](atom)
        out.extend(_one_hot_featurizer(idx, length))
    return out


def get_bond_features(bond: Bond) -> list[bool]:
    assert Vocabulary.BOND_VOCAB is not None
    out = []
    for func_name in Vocabulary.BOND_VOCAB:
        idx, length = BOND_FUNCTIONS[func_name](bond)
        out.extend(_one_hot_featurizer(idx, length))
    return out


ATOM_FUNCTIONS = {
    "element": get_atom_element_idx,
    "degree": get_atom_degree_idx,
    "formal_charge": get_atom_formal_charge_idx,
    "hybridization": get_atom_hybridization_idx,
    "chirality": get_atom_chirality_idx,
    "hydrogens": get_atom_hydrogens_idx,
    "aromaticity": get_atom_aromaticity_idx,
}

BOND_FUNCTIONS = {
    "order": get_bond_order_idx,
    "in_ring": get_bond_ring_idx,
    "conjugated": get_bond_conjugated_idx,
    "stereo": get_bond_stereo_idx,
}

DEFAULT_ATOM_VOCAB = ["element", "degree", "formal_charge"]

DEFAULT_BOND_VOCAB = ["order"]


class Vocabulary:
    """Class to manage atom and edge featurizations"""

    __MOL = Chem.MolFromSmiles("CC")
    NUM_NODE_FEATURES = -1
    NUM_EDGE_FEATURES = -1
    ATOM_VOCAB = None
    BOND_VOCAB = None

    @staticmethod
    def register_atom_vocab(vocab: Optional[list[str]] = None):
        Vocabulary.ATOM_VOCAB = sorted(
            DEFAULT_ATOM_VOCAB if vocab is None else vocab
        )
        Vocabulary.NUM_NODE_FEATURES = len(
            get_atom_features(Vocabulary.__MOL.GetAtoms()[0])
        )
        logging.debug(
            f"Registered atom features: [{', '.join(Vocabulary.ATOM_VOCAB)}]"
        )

    @staticmethod
    def register_bond_vocab(vocab: Optional[list[str]] = None):
        Vocabulary.BOND_VOCAB = sorted(
            DEFAULT_BOND_VOCAB if vocab is None else vocab
        )
        Vocabulary.NUM_EDGE_FEATURES = len(
            get_bond_features(Vocabulary.__MOL.GetBonds()[0])
        )
        logging.debug(
            f"Registered edge features: [{', '.join(Vocabulary.BOND_VOCAB)}]"
        )

    @staticmethod
    def atom_feature_in_vocab(key):
        return key in Vocabulary.ATOM_VOCAB

    def bond_feature_in_vocab(key):
        return key in Vocabulary.BOND_VOCAB

    @staticmethod
    def get_num_node_features() -> int:
        assert Vocabulary.NUM_NODE_FEATURES > 0
        return Vocabulary.NUM_NODE_FEATURES

    @staticmethod
    def get_num_edge_features() -> int:
        assert Vocabulary.NUM_EDGE_FEATURES > 0
        return Vocabulary.NUM_EDGE_FEATURES
