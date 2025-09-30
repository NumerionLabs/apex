# Standard
import logging
import sys
from typing import Iterator, Optional

# Third party
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AllChem import ReactionFromSmarts, ReactionToSmarts
from rdkit.Chem.rdchem import AtomValenceException, KekulizeException
import torch
from torch.utils.data import DataLoader, Dataset

# Atomwise
from apex.data.atom_bond_features import (
    get_sticky_smiles,
    Vocabulary,
)
from apex.data.torch_graph import PackedTorchMol, TorchMol
from apex.utils.other_utils import flatten_list, int2mix
from apex.utils.torch_utils import (
    build_library_indexes,
    convert_1d_to_2d_indexes,
)


class CSLDataset(Dataset):
    def __init__(
        self,
        reaction_df: pd.DataFrame,
        synthon_df: pd.DataFrame,
        fast_smiles: bool = True,
        synthon_torch_mol: bool = True,
        load_synthons_and_reactions: bool = False,
    ):
        """
        PyTorch dataset for Combinatorial Synthesis Libraries.
        Args:
            reaction_df: Dataframe with reaction SMARTS
            synthon_df: Dataframe with synthon SMILES
            fast_smiles: Whether to use string-based construction of product
              SMILES
            synthon_torch_mol: Whether the __getitem__ should include synthons
              as TorchMol objects
            load_synthons_and_reactions: Whether to load the synthons and
              reactions with RDKit, primarily used for speeding up methods like
              key2smiles when fast_smiles is False
        """
        # Copy dataframes
        self._orig_reaction_df = reaction_df
        self._orig_synthon_df = synthon_df
        reaction_df = self._orig_reaction_df.copy()
        synthon_df = self._orig_synthon_df.copy()

        # Should SMILES be constructed via regex or via an RDKit reaction
        self.fast_smiles = bool(fast_smiles)

        # Should the synthon molecular graphs be computed
        self.synthon_torch_mol = synthon_torch_mol

        # Should the synthons and reactions be loaded
        self.load_synthons_and_reactions = load_synthons_and_reactions

        # Map the original reaction_ids to the internal normalized reaction_ids
        # (which is like range(n_reactions))
        reaction_df["orig_reaction_id"] = reaction_df["reaction_id"]
        orig_reaction_mapper = {
            name: i for i, name in enumerate(reaction_df["reaction_id"])
        }
        reaction_df["reaction_id"] = reaction_df["orig_reaction_id"].apply(
            lambda x: orig_reaction_mapper[x]
        )

        # Map the original synthon_ids to the internal normalized synthon_ids
        # (which is like range(n_synthons))
        synthon_df["orig_synthon_id"] = synthon_df["synthon_id"]
        synthon_mapper = {
            smi: i
            for i, smi in enumerate(
                sorted(synthon_df["smiles"].unique())
            )
        }
        synthon_df["synthon_id"] = synthon_df["smiles"].apply(
            lambda x: synthon_mapper[x]
        )
        synthon_df["orig_reaction_id"] = synthon_df["reaction_id"]
        synthon_df["reaction_id"] = synthon_df["orig_reaction_id"].apply(
            lambda x: orig_reaction_mapper[x]
        )

        # Form dicts for mapping internal reaction/synthon IDs to originals
        self._orig_reaction_id_lookup = {
            reaction_id: orig_reaction_id
            for reaction_id, orig_reaction_id in zip(
                reaction_df.reaction_id,
                reaction_df.orig_reaction_id,
            )
        }
        self._orig_synthon_id_lookup = {
            (reaction_id, synthon_id): orig_synthon_id
            for reaction_id, synthon_id, orig_synthon_id in zip(
                synthon_df.reaction_id,
                synthon_df.synthon_id,
                synthon_df.orig_synthon_id,
            )
        }

        # Form libtree; this is a list-of-list-of-list-of-ints. The top level
        # corresponds to reactions (i.e., len(libtree) == n_reactions), with
        # the subsequent level corresponding to the reaction R-groups, and the
        # leaves are the normalized synthon_ids (ints) for the synthons
        # contained in the given R-group
        libtree = synthon_df[["reaction_id", "rgroup", "synthon_id"]]
        libtree = libtree.drop_duplicates()
        libtree = libtree.sort_values(
            by=["reaction_id", "rgroup", "synthon_id"]
        ).reset_index(drop=True)
        libtree = libtree.groupby("reaction_id")
        libtree = libtree.apply(
            lambda x: x.groupby("rgroup").apply(
                lambda x: x["synthon_id"].tolist()
            )
        ).T
        libtree: list[list[list[int]]] = [
            libtree[i].tolist() for i in range(len(orig_reaction_mapper))
        ]

        assert len(set(orig_reaction_mapper.values())) == len(
            set(orig_reaction_mapper.keys())
        )
        tmp = synthon_df[
            ["orig_synthon_id", "smiles"]
        ].drop_duplicates()
        assert len(tmp.orig_synthon_id.unique()) == len(tmp)
        orig_synthon_mapper = {
            k: synthon_mapper[v]
            for k, v in zip(tmp.orig_synthon_id, tmp.synthon_smiles)
        }

        # Retain only specific columns
        reaction_df = reaction_df[
            ["reaction_id", "orig_reaction_id", "smarts"]
        ].sort_values(by="reaction_id")
        synthon_df = synthon_df[
            ["synthon_id", "orig_synthon_id", "smiles"]
        ].sort_values(by="synthon_id")

        # Create a bunch of attributes
        self.reaction_df: pd.DataFrame = (
            reaction_df.drop_duplicates().reset_index(drop=True)
        )
        self.synthon_df: pd.DataFrame = (
            synthon_df.drop_duplicates().reset_index(drop=True)
        )
        self.libtree: list[list[list[int]]] = libtree
        self.reaction_smarts: list[str] = (
            self.reaction_df[["reaction_id", "smarts"]]
            .drop_duplicates()["smarts"]
            .tolist()
        )
        self.synthon_smiles: list[str] = (
            self.synthon_df[["synthon_id", "smiles"]]
            .drop_duplicates()["smiles"]
            .tolist()
        )
        self.sticky_synthon_smiles: list[str] = [
            get_sticky_smiles(smi) for smi in self.synthon_smiles
        ]
        self._rgroup_counts = [
            [len(x) for x in v] for k, v in enumerate(self.libtree)
        ]
        self._reaction_counts = np.array(
            [np.prod(self._rgroup_counts[k]) for k in range(self.num_reactions)]
        )
        self._reaction_counts_cum = np.insert(
            np.cumsum(self._reaction_counts), 0, 0
        )
        self._orig_synthon_mapper = orig_synthon_mapper
        self._orig_reaction_mapper = orig_reaction_mapper
        self._num_products = sum(
            [np.prod([len(y) for y in x]) for x in self.libtree]
        )
        self._num_rgroups = sum([len(x) for x in self._rgroup_counts])

        # Load synthons and reactions
        self._enumeration_mode = False
        self.synthon_mols = None
        self.reaction_rxns = None
        if self.load_synthons_and_reactions:
            self.synthon_mols = [
                self.synthon2mol(i) for i in range(self.num_synthons)
            ]
            self.reaction_rxns = [
                self.reaction2rxn(i) for i in range(self.num_reactions)
            ]

    def get_internal_synthon_id(self, orig_synthon_id) -> int:
        return self._orig_synthon_mapper[orig_synthon_id]

    def get_internal_reaction_id(self, orig_reaction_id) -> int:
        return self._orig_reaction_mapper[orig_reaction_id]

    @property
    def num_node_features(self) -> int:
        return Vocabulary.get_num_node_features()

    @property
    def num_edge_features(self) -> int:
        return Vocabulary.get_num_edge_features()

    @property
    def num_reactions(self) -> int:
        return len(self.reaction_smarts)

    @property
    def num_rgroups(self) -> int:
        return self._num_rgroups

    @property
    def num_synthons(self) -> int:
        return len(self.synthon_smiles)

    @property
    def num_products(self) -> int:
        return self._num_products

    def get_synthons_smiles(self) -> list[str]:
        return self.synthon_smiles.copy()

    def get_product_ids_by_reaction_id(self, reaction_id: int) -> range:
        return range(
            self._reaction_counts_cum[reaction_id],
            self._reaction_counts_cum[reaction_id + 1],
        )

    def idx2key(self, idx: int) -> tuple[int, tuple[int, ...]]:
        assert 0 <= idx < len(self)
        reaction_id = (self._reaction_counts_cum <= idx).sum() - 1
        mix = int2mix(
            idx - self._reaction_counts_cum[reaction_id],
            self._rgroup_counts[reaction_id],
        )
        synthon_ids = tuple(
            x[i] for i, x in zip(mix, self.libtree[reaction_id])
        )
        return reaction_id, synthon_ids

    def key2smiles(self, reaction_id: int, synthon_ids: tuple[int, ...]) -> str:
        if self.fast_smiles:
            synthons = [
                self.synthon2smiles(synthon_id).split(".")[0]
                for synthon_id in synthon_ids
            ]
            return ".".join(synthons)
        else:
            return Chem.MolToSmiles(self.key2mol(reaction_id, synthon_ids))

    def key2mol(
        self, reaction_id: int, synthon_ids: tuple[int, ...]
    ) -> Chem.rdchem.Mol:
        if self.fast_smiles:
            return Chem.MolFromSmiles(self.key2smiles(reaction_id, synthon_ids))
        else:
            if self.load_synthons_and_reactions:
                reaction = self.reaction_rxns[reaction_id]
                synthons = [self.synthon_mols[i] for i in synthon_ids]
                return reaction.RunReactants(synthons)[0][0]

            reaction = self.reaction2rxn(reaction_id)
            synthons = tuple(self.synthon2mol(i) for i in synthon_ids)
            try:
                products = reaction.RunReactants(synthons)
            except Exception as e:
                rxn_smarts = ReactionToSmarts(reaction)
                synthon_smiles = [
                    Chem.MolToSmiles(synthon) for synthon in synthons
                ]
                raise RuntimeError(
                    f"key2smiles {rxn_smarts} {synthon_smiles} failed: {e}"
                )
            product = products[0][0]
            return product

    def key2name(self, reaction_id: int, synthon_ids: tuple[int, ...]) -> str:
        sep = "____"
        orig_reaction_id = self._orig_reaction_id_lookup[reaction_id]
        orig_synthon_ids = []
        for synthon_id in synthon_ids:
            orig_synthon_id = self._orig_synthon_id_lookup[
                (reaction_id, synthon_id)
            ]
            orig_synthon_ids.append(str(orig_synthon_id))
        return str(orig_reaction_id) + sep + sep.join(orig_synthon_ids)

    def name2key(self, name: str) -> tuple[int, tuple[int, ...]]:
        sep = "____"
        split_name = name.split(sep)
        orig_reaction_id = split_name[0]
        orig_synthon_ids = tuple(split_name[1:])
        reaction_id = self.get_internal_reaction_id(orig_reaction_id)
        synthon_ids = tuple(
            self.get_internal_synthon_id(x) for x in orig_synthon_ids
        )
        return (reaction_id, synthon_ids)

    def reaction2smarts(self, reaction_id: int) -> str:
        return self.reaction_smarts[reaction_id]

    def reaction2rxn(
        self, reaction_id: int
    ) -> Chem.rdChemReactions.ChemicalReaction:
        return ReactionFromSmarts(self.reaction2smarts(reaction_id))

    def synthon2smiles(self, synthon_id: int) -> str:
        if self.fast_smiles:
            return self.sticky_synthon_smiles[synthon_id]
        else:
            return self.synthon_smiles[synthon_id]

    def synthon2mol(self, synthon_id: int) -> Chem.rdchem.Mol:
        return Chem.MolFromSmiles(self.synthon2smiles(synthon_id))

    def __len__(self) -> int:
        return self.num_products

    def __getitem__(self, product_id: int):
        (reaction_id, synthon_ids) = self.idx2key(product_id)
        try:
            if self.fast_smiles:
                smiles = self.key2smiles(reaction_id, synthon_ids)
                mol = Chem.MolFromSmiles(smiles)
            else:
                mol = self.key2mol(reaction_id, synthon_ids)
                smiles = Chem.MolToSmiles(mol)
            if self._enumeration_mode:
                return {
                    "smiles": smiles,
                    "name": self.key2name(reaction_id, synthon_ids),
                }
            tmol = TorchMol(mol)
        except (KekulizeException, AtomValenceException) as e:
            logging.debug(f"product_id: {product_id}")
            logging.debug(f"reaction_id: {reaction_id}")
            logging.debug(f"synthon_ids: {synthon_ids}")
            logging.debug(f"Error: {e}")
            return None
        except IndexError as e:
            logging.debug(f"product_id: {product_id}")
            logging.debug(f"reaction_id: {reaction_id}")
            logging.debug(f"synthon_ids: {synthon_ids}")
            logging.debug(f"Error: {e}")
            return None

        return {
            "smiles": smiles,
            "key": (reaction_id, synthon_ids),
            "product_id": product_id,
            "reaction_id": reaction_id,
            "synthon_ids": synthon_ids,
            "product": tmol,
            "synthons": (
                [TorchMol(self.synthon2mol(i)) for i in synthon_ids]
                if self.synthon_torch_mol
                else []
            ),
        }

    @staticmethod
    def collate_fn(items):
        items = [item for item in items if item is not None]
        orig_product_ids = [item["product_id"] for item in items]

        libtree = {}
        for item in items:
            (reaction_id, synthon_ids) = (
                item["reaction_id"],
                item["synthon_ids"],
            )
            if reaction_id not in libtree:
                libtree[reaction_id] = [{s} for s in synthon_ids]
            else:
                for i in range(len(libtree[reaction_id])):
                    libtree[reaction_id][i].update({synthon_ids[i]})

        orig_reaction_ids = sorted(libtree.keys())
        libtree = [list(map(sorted, libtree[k])) for k in orig_reaction_ids]

        library_indexes = build_library_indexes(libtree)

        reaction_mapper = {j: i for i, j in enumerate(orig_reaction_ids)}
        synthon_mapper = {
            j.item(): i
            for i, j in enumerate(library_indexes["orig_synthon_ids"])
        }

        orig_keys = []
        keys = []
        for item in items:
            reaction_id = item["reaction_id"]
            synthon_ids = item["synthon_ids"]
            orig_key = (reaction_id, synthon_ids)
            key = (
                reaction_mapper[reaction_id],
                tuple(synthon_mapper[i] for i in synthon_ids),
            )
            orig_keys.append(orig_key)
            keys.append(key)

        product2reaction = []
        block2product = []
        block2rgcount = []
        block2synthon = []
        for i, (reaction_id, synthon_ids) in enumerate(keys):
            product2reaction.append(reaction_id)
            block2product.extend(len(synthon_ids) * [i])
            block2rgcount.extend(list(range(len(synthon_ids))))
            block2synthon.extend([synthon_id for synthon_id in synthon_ids])

        product2reaction = convert_1d_to_2d_indexes(
            torch.tensor(product2reaction)
        )
        block2product = convert_1d_to_2d_indexes(torch.tensor(block2product))
        block2rgcount = convert_1d_to_2d_indexes(torch.tensor(block2rgcount))
        block2synthon = convert_1d_to_2d_indexes(torch.tensor(block2synthon))
        block2reaction = convert_1d_to_2d_indexes(
            product2reaction[1][block2product[1]]
        )
        block2rgroup = convert_1d_to_2d_indexes(
            block2rgcount[1]
            + library_indexes["first_rgroup_by_reaction"][block2reaction[1]]
        )

        (idx0, idx1) = block2synthon[:, block2synthon[1].argsort()]
        idx2 = idx0[
            torch.where(torch.nn.functional.pad(idx1.diff(), (1, 0), value=1))[
                0
            ]
        ]

        products = [item["product"] for item in items]

        blocks = flatten_list([item["synthons"] for item in items])
        if blocks:
            synthons = [blocks[i.item()] for i in idx2]
        else:
            synthons = None

        return {
            "keys": keys,
            "orig_keys": orig_keys,
            "orig_reaction_ids": torch.tensor(
                orig_reaction_ids, dtype=torch.long
            ),
            "orig_product_ids": torch.tensor(
                orig_product_ids, dtype=torch.long
            ),
            "library_indexes": library_indexes,
            "product2reaction": product2reaction,
            "block2product": block2product,
            "block2reaction": block2reaction,
            "block2rgroup": block2rgroup,
            "block2synthon": block2synthon,
            "products": PackedTorchMol(products),
            "synthons": PackedTorchMol(synthons) if synthons else None,
        }

    def create_dataloader_uniformly_at_random(
        self, num_products: int, max_iterations: Optional[int] = None
    ) -> DataLoader:
        self._enumeration_mode = False

        class UniformRandomBatchSampler:
            def __init__(self_, dataset, num_products, max_iterations):
                self_.dataset = dataset
                self_.num_products = int(num_products)
                self_.max_iterations = int(max_iterations or sys.maxsize)
                self_.cum_iterations = 0

            def __iter__(self_) -> Iterator[list[int]]:
                self_.cum_iterations = 0
                while self_.cum_iterations < self_.max_iterations:
                    self_.cum_iterations += 1
                    indexes = np.random.randint(
                        0, len(self_.dataset), (self_.num_products,)
                    ).tolist()
                    yield indexes

        return DataLoader(
            dataset=self,
            batch_sampler=UniformRandomBatchSampler(
                dataset=self,
                num_products=num_products,
                max_iterations=max_iterations,
            ),
            collate_fn=self.collate_fn,
        )

    def create_dataloader_conditionally_uniformly_at_random(
        self,
        num_products: int,
        keep_reactions: Optional[list[int]] = None,
        max_iterations: Optional[int] = None,
    ) -> DataLoader:
        self._enumeration_mode = False

        class ConditionallyUniformRandomBatchSampler:
            def __init__(
                self_, dataset, num_products, keep_reactions, max_iterations
            ):
                self_.dataset = dataset
                self_.num_products = int(num_products)
                self_.keep_reactions = keep_reactions
                self_.max_iterations = int(max_iterations or sys.maxsize)
                self_.cum_iterations = 0

            def __iter__(self_) -> Iterator[list[int]]:
                self_.cum_iterations = 0
                while self_.cum_iterations < self_.max_iterations:
                    self_.cum_iterations += 1
                    if self_.keep_reactions is not None:
                        reaction_ids = np.random.choice(
                            keep_reactions, self_.num_products
                        )
                    else:
                        reaction_ids = np.random.randint(
                            0,
                            self_.dataset.num_reactions,
                            (self_.num_products,),
                        )
                    ranges = [
                        self_.dataset.get_product_ids_by_reaction_id(i)
                        for i in reaction_ids
                    ]
                    indexes = [
                        np.random.randint(rng.start, rng.stop) for rng in ranges
                    ]
                    yield indexes

        return DataLoader(
            dataset=self,
            batch_sampler=ConditionallyUniformRandomBatchSampler(
                dataset=self,
                num_products=num_products,
                keep_reactions=keep_reactions,
                max_iterations=max_iterations,
            ),
            collate_fn=self.collate_fn,
        )

    def create_dataloader_stratified_uniformly_at_random(
        self,
        num_products_per_reaction: int,
        num_reactions: int = 1,
        max_iterations: Optional[int] = None,
    ) -> DataLoader:
        self._enumeration_mode = False

        class StratifiedUniformRandomBatchSampler:
            def __init__(
                self_,
                dataset,
                num_products_per_reaction,
                num_reactions,
                max_iterations,
            ):
                self_.dataset = dataset
                self_.num_products_per_reaction = int(num_products_per_reaction)
                self_.num_reactions = int(num_reactions)
                self_.max_iterations = int(max_iterations or sys.maxsize)
                self_.cum_iterations = 0

            def __iter__(self_) -> Iterator[list[int]]:
                self_.cum_iterations = 0
                while self_.cum_iterations < self_.max_iterations:
                    self_.cum_iterations += 1
                    reaction_ids = np.random.choice(
                        range(self_.dataset.num_reactions),
                        self_.num_reactions,
                        replace=False,
                    ).tolist()
                    ranges = [
                        self_.dataset.get_product_ids_by_reaction_id(i)
                        for i in reaction_ids
                    ]
                    indexes = flatten_list(
                        [self_.random_sample(rng) for rng in ranges]
                    )
                    yield indexes

            def random_sample(self_, rng):
                return np.random.randint(
                    rng.start, rng.stop, self_.num_products_per_reaction
                ).tolist()

        return DataLoader(
            dataset=self,
            batch_sampler=StratifiedUniformRandomBatchSampler(
                dataset=self,
                num_products_per_reaction=num_products_per_reaction,
                num_reactions=num_reactions,
                max_iterations=max_iterations,
            ),
            collate_fn=self.collate_fn,
        )

    def sample_random_subset(self, min_size: int) -> "CSLDataset":
        """
        Samples a random CSLDataset subset comprised of at least `min_size`
        compounds. In going from the original CSLDataset to the random subset,
        the relative sizes of the reactions are preserved, as are the relative
        sizes of the R-groups (i.e., synthon counts) in a reaction.
        """
        rgroup_sizes = [[len(x) for x in y] for y in self.libtree]
        reaction_sizes = np.array(list(map(np.prod, rgroup_sizes)))
        reaction_fracs = reaction_sizes / reaction_sizes.sum()
        new_reaction_sizes = np.ceil(reaction_fracs * min_size).astype(int)
        new_rgroup_sizes = []
        for i in range(len(reaction_sizes)):
            synthon_fracs = np.array(rgroup_sizes[i]) / sum(rgroup_sizes[i])
            rescale = (new_reaction_sizes[i] / np.prod(synthon_fracs)) ** (
                1 / len(rgroup_sizes[i])
            ) * synthon_fracs
            new_rgroup_sizes.append(np.ceil(rescale).astype(int).tolist())

        df_parts = []
        for i, rgroup_size in enumerate(new_rgroup_sizes):
            for j, c in enumerate(rgroup_size):
                df_part = self._orig_synthon_df.loc[
                    (
                        self._orig_synthon_df["reaction_id"]
                        == self._orig_reaction_id_lookup[i]
                    )
                    & (self._orig_synthon_df["synton#"] == j + 1)
                ].sample(c)
                df_parts.append(df_part)

        return CSLDataset(
            reaction_df=self._orig_reaction_df,
            synthon_df=pd.concat(df_parts).reset_index(drop=True),
            synthon_torch_mol=self.synthon_torch_mol,
            load_synthons_and_reactions=self.load_synthons_and_reactions,
        )

    def get_input_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (self._orig_reaction_df, self._orig_synthon_df)

    def enumerate(self, **kwargs) -> pd.DataFrame:
        def collate_fn(items):
            return {
                "name": [item["name"] for item in items],
                "smiles": [item["smiles"] for item in items],
            }

        self._enumeration_mode = True
        batch_size = int(kwargs.get("batch_size", 10_000))
        logging_iterations = int(kwargs.get("logging_iterations", 100))
        (name_list, smiles_list) = ([], [])
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        for batch_iter, batch in enumerate(dataloader):
            name_list.extend(batch["name"])
            smiles_list.extend(batch["smiles"])
            if batch_iter % logging_iterations == 0:
                logging.info(batch_iter, len(name_list), len(smiles_list))
        self._enumeration_mode = False
        return pd.DataFrame({"name": name_list, "smiles": smiles_list})
