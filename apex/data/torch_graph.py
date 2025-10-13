# Standard
from typing import Optional

# Third party
from rdkit import Chem
import torch
from torch import LongTensor, Tensor

# APEX
from apex.data.atom_bond_features import (
    get_atom_features,
    get_bond_features,
    prepare_molecule,
)


class TorchGraph:
    node_features: Tensor
    edge_features: Tensor
    edge_index: LongTensor
    graph_index: LongTensor

    @property
    def params(self) -> tuple[Tensor, Tensor, LongTensor, LongTensor]:
        return (
            self.node_features,
            self.edge_features,
            self.edge_index,
            self.graph_index,
        )

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_features.size(0)

    @property
    def num_node_features(self) -> int:
        return self.node_features.size(1)

    @property
    def num_edge_features(self) -> int:
        return self.edge_features.size(1)

    @property
    def num_graphs(self) -> int:
        raise NotImplementedError

    @property
    def device(self):
        devices = list(set([p.device for p in self.params]))
        assert len(devices) == 1
        return devices[0]

    def to(self, device):
        self.node_features = self.node_features.to(device)
        self.edge_features = self.edge_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.graph_index = self.graph_index.to(device)
        return self

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int):
        raise NotImplementedError


class TorchMol(TorchGraph):
    def __init__(
        self,
        mol: Chem.Mol,
        prepare_mol: bool = True,
        remove_hydrogens: bool = True,
    ):
        super().__init__()
        self.mol = mol
        if prepare_mol:
            self.mol = prepare_molecule(self.mol, remove_hydrogens)
        (
            self.node_features,
            self.edge_features,
            self.edge_index,
            self.graph_index,
        ) = self.process()

    def process(self) -> tuple[Tensor, Tensor, LongTensor, LongTensor]:
        # Get node and edge features, as well as edge indexes
        node_features: list[list[bool]] = [
            get_atom_features(atom) for atom in self.mol.GetAtoms()
        ]
        edge_features: list[list[bool]] = [
            get_bond_features(bond) for bond in self.mol.GetBonds()
        ]
        edge_index: list[list[int]] = [
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            for bond in self.mol.GetBonds()
        ]

        # Make them tensors
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # Make edges bidirectional
        edge_features = edge_features.repeat(2, 1)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], 1)

        # Graph index
        graph_index = torch.zeros((node_features.size(0),), dtype=torch.long)

        return node_features, edge_features, edge_index, graph_index

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        remove_hydrogens: bool = True,
        sanitize: bool = True,
    ):
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        return cls(mol, remove_hydrogens)

    @classmethod
    def from_smarts(
        cls,
        smarts: str,
        remove_hydrogens: bool = False,
    ):
        mol = Chem.MolFromSmarts(smarts)
        if mol.NeedsUpdatePropertyCache():
            mol.UpdatePropertyCache(strict=False)
        return cls(mol, remove_hydrogens)

    @property
    def num_graphs(self):
        return 1

    def __getitem__(self, idx: int):
        assert idx == 0
        return self

    def view(self):
        return Chem.Draw.MolsToGridImage(
            mols=[self.mol], molsPerRow=1, maxMols=1, legends=["0"], useSVG=True
        )


class PackedTorchMol(TorchGraph):
    def __init__(self, torch_mols: list[TorchMol]):
        super().__init__()
        self.torch_mols: list[TorchMol] = torch_mols
        (
            self.node_features,
            self.edge_features,
            self.edge_index,
            self.graph_index,
        ) = self.process()

    @classmethod
    def from_smiles(
        cls,
        smiles_list: list[str],
        remove_hydrogens: bool = True,
        sanitize: bool = True,
    ):
        torch_mols = [
            TorchMol.from_smiles(smiles, remove_hydrogens, sanitize)
            for smiles in smiles_list
        ]
        return cls(torch_mols)

    @classmethod
    def from_smarts(
        cls,
        smarts_list: list[str],
        remove_hydrogens: bool = False,
    ):
        torch_mols = [
            TorchMol.from_smarts(smarts, remove_hydrogens)
            for smarts in smarts_list
        ]
        return cls(torch_mols)

    def process(self) -> tuple[Tensor, Tensor, LongTensor, LongTensor]:
        num_nodes = [tm.num_nodes for tm in self.torch_mols]
        cum_num_nodes = torch.tensor([0] + num_nodes).cumsum(0)[:-1]
        node_features = torch.cat(
            [tm.node_features for tm in self.torch_mols], 0
        )
        edge_index = torch.cat(
            [
                tm.edge_index + i
                for i, tm in zip(cum_num_nodes, self.torch_mols)
            ],
            1,
        )
        edge_features = torch.cat(
            [tm.edge_features for tm in self.torch_mols], 0
        )
        graph_index = torch.cat(
            [
                i + torch.zeros_like(tm.graph_index)
                for i, tm in enumerate(self.torch_mols)
            ],
            0,
        )
        return node_features, edge_features, edge_index, graph_index

    @property
    def num_graphs(self):
        return len(self.torch_mols)

    def __getitem__(self, idx: int):
        assert idx >= 0
        return self.torch_mols[idx]

    def view(self, mols_per_row: Optional[int] = None, max_mols: int = 100):
        return Chem.Draw.MolsToGridImage(
            mols=[tm.mol for tm in self.torch_mols],
            molsPerRow=mols_per_row or len(self),
            maxMols=max_mols,
            legends=list(map(str, range(len(self)))),
            useSVG=True,
        )
