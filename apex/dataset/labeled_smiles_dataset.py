# Standard
import logging
import sys
from typing import Iterator, Optional

# Third party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Atomwise
from apex.data.atom_bond_features import Vocabulary
from apex.data.torch_graph import PackedTorchMol, TorchMol

logger = logging.getLogger(__name__)


class LabeledSmilesDataset(Dataset):
    """
    PyTorch dataset for a numerically labeled SMILES dataframe.

    Args:
        df: Dataframe with labeled SMILES data.
        property_columns: Names of columns with numerical properties.
        smiles_column: Name of column with SMILES.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        property_columns: list[str] = [],
        smiles_column: str = "smiles",
    ):
        super().__init__()

        self.property_columns = property_columns
        self.smiles_column = smiles_column

        idx = df[self.property_columns].notna().all(axis=1)
        self.df = df.loc[idx].reset_index(drop=True)

    @property
    def num_node_features(self) -> int:
        return Vocabulary.NUM_NODE_FEATURES

    @property
    def num_edge_features(self) -> int:
        return Vocabulary.NUM_EDGE_FEATURES

    @property
    def num_properties(self):
        return len(self.property_columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            smiles = row[self.smiles_column]
            mol = TorchMol.from_smiles(row[self.smiles_column])
            values = torch.tensor(
                row.loc[self.property_columns].tolist(),
                dtype=torch.float,
            )
            return {"mol": mol, "values": values}
        except Exception as e:
            logger.warning(f"idx {idx}, smiles {smiles} failed: {e}")
            return None

    @staticmethod
    def collate_fn(items):
        items = [item for item in items if item is not None]
        mols = PackedTorchMol([item["mol"] for item in items])
        values = torch.stack([item["values"] for item in items])
        return {"mols": mols, "values": values}

    def create_dataloader(
        self,
        batch_size: int,
        max_iterations: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        class UniformRandomBatchSampler:
            def __init__(self_, dataset, batch_size, max_iterations):
                self_.dataset = dataset
                self_.batch_size = int(batch_size)
                self_.max_iterations = int(max_iterations or sys.maxsize)
                self_.cum_iterations = 0

            def __iter__(self_) -> Iterator[list[int]]:
                self_.cum_iterations = 0
                while self_.cum_iterations < self_.max_iterations:
                    self_.cum_iterations += 1
                    indexes = np.random.randint(
                        0, len(self_.dataset), (self_.batch_size,)
                    ).tolist()
                    yield indexes

        return torch.utils.data.DataLoader(
            dataset=self,
            batch_sampler=UniformRandomBatchSampler(
                dataset=self,
                batch_size=batch_size,
                max_iterations=max_iterations,
            ),
            collate_fn=self.collate_fn,
        )
