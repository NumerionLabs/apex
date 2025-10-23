# Standard
from functools import partial
import logging
import math
import os
import time
from typing import ClassVar, Optional

# Third party
import pandas as pd
import torch
from torch import LongTensor, nn, Tensor
from torch.utils.tensorboard import SummaryWriter

# APEX
from apex_topk import topk
from apex.dataset.csl_dataset import CSLDataset
from apex.dataset.labeled_smiles_dataset import LabeledSmilesDataset
from apex.nn.encoder import LigandEncoder
from apex.nn.mlp import ResNetMLP
from apex.nn.probe import LinearProbe
from apex.utils.torch_utils import build_library_indexes
from apex.nn.scatter import Scatter

DEFAULT_CHUNK_SIZE: int = 2**30

logger = logging.getLogger(__name__)


class APEXFactorizer(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        associative_dim: int,
        hidden_dim: int,
        num_layers: int,
        mpnn_params: dict,
        normalization: bool = True,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.associative_dim = int(associative_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.mpnn_params = dict(mpnn_params)
        self.normalization = bool(normalization)

        self.synthon_graph_encoder = LigandEncoder(
            embed_dim=self.hidden_dim,
            mpnn_params=self.mpnn_params,
        )

        self.rgroup_encoder_p0 = ResNetMLP(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            normalization=self.normalization,
        )

        self.rgroup_encoder_p1 = ResNetMLP(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            normalization=self.normalization,
        )

        self.rgroup_encoder_pooling_function = Scatter(
            op="mean",
            unbind_index=True,
            dim=0,
        )

        self.reaction_encoder_p0 = ResNetMLP(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            normalization=self.normalization,
        )

        self.reaction_encoder_p1 = ResNetMLP(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            normalization=self.normalization,
        )

        self.reaction_encoder_pooling_function = Scatter(
            op="add",
            unbind_index=True,
            dim=0,
        )

        self.rgroup_associative_key_encoder = ResNetMLP(
            input_dim=self.hidden_dim,
            output_dim=self.embed_dim * self.associative_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            normalization=self.normalization,
        )

        self.synthon_associative_value_encoder = ResNetMLP(
            input_dim=self.hidden_dim,
            output_dim=self.associative_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            normalization=self.normalization,
        )

        self.embed_predictor_pooling_function = Scatter(
            op="add",
            unbind_index=True,
            dim=0,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "APEXFactorizer":
        state_dict = torch.load(path, **kwargs)
        model = cls(
            embed_dim=state_dict["embed_dim"],
            associative_dim=state_dict["associative_dim"],
            hidden_dim=state_dict["hidden_dim"],
            num_layers=state_dict["num_layers"],
            mpnn_params=state_dict["mpnn_params"],
            normalization=state_dict["normalization"],
        )
        model.load_state_dict(state_dict["model_state_dict"])
        return model

    def save(self, path: str):
        state_dict = {
            "model_state_dict": self.state_dict(),
            "embed_dim": self.embed_dim,
            "associative_dim": self.associative_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "mpnn_params": self.mpnn_params,
            "normalization": self.normalization,
        }
        torch.save(state_dict, path)


class APEXFactorizedCSL(nn.Module):

    ITER_STATE_KEY: ClassVar[str] = "iter"
    MODEL_STATE_KEY: ClassVar[str] = "model_state_dict"
    OPTIMIZER_STATE_KEY: ClassVar[str] = "optimizer_state_dict"
    LOSS_STATE_KEY: ClassVar[str] = "loss"

    def __init__(
        self,
        encoder: LigandEncoder,
        factorizer: APEXFactorizer,
        dataset: CSLDataset,
        probe: Optional[LinearProbe] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.factorizer = factorizer
        self.dataset = dataset
        assert self.encoder.embed_dim == self.factorizer.embed_dim
        self.embed_dim = int(self.factorizer.embed_dim)
        self.associative_dim = int(self.factorizer.associative_dim)

        device = next(self.encoder.parameters()).device
        self.factorizer.to(device)

        if probe is not None:
            self.probe = probe.to(device)

        num_asynthon = sum(sum(len(x) for x in y) for y in self.dataset.libtree)

        make_zeros = partial(torch.zeros, dtype=torch.float, device=device)
        library_tensors = [
            (
                "synthon_associative_embeds",
                make_zeros(num_asynthon, self.associative_dim) / 0,
            ),
            (
                "synthon_feats",
                make_zeros(self.dataset.num_synthons, self.embed_dim) / 0,
            ),
            (
                "rgroup_feats",
                make_zeros(self.dataset.num_rgroups, self.embed_dim) / 0,
            ),
            (
                "reaction_feats",
                make_zeros(self.dataset.num_reactions, self.embed_dim) / 0,
            ),
        ]
        if self.probe is not None:
            library_tensors.append(
                (
                    "synthon_associative_contribs",
                    make_zeros(num_asynthon, probe.output_dim) / 0,
                )
            )

        self.library_tensors = nn.ParameterDict(
            {
                k: nn.Parameter(v, requires_grad=False)
                for k, v in library_tensors
            }
        )
        self.library_indexes = nn.ParameterDict(
            {
                k: nn.Parameter(v, requires_grad=False)
                for k, v in build_library_indexes(
                    self.dataset.libtree, device
                ).items()
            }
        )

        self.asynthon_mapper = {
            tuple(idx.tolist()): i
            for i, idx in enumerate(self.library_indexes.synthon2rgroup.T)
        }
        self.apex_libtree = self.build_apex_libtree()

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, other: LigandEncoder):
        if other.embed_dim != self.factorizer.embed_dim:
            raise RuntimeError(
                "LigandEcoder does not match existing embedding dimension"
            )

        self._encoder = other

    def predict_embeds_from_keys(
        self,
        keys: list[tuple[int, tuple[int, ...]]],
        synthon_associative_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        device = next(self.factorizer.parameters()).device
        if synthon_associative_embeds is None:
            synthon_associative_embeds = (
                self.library_tensors.synthon_associative_embeds
            )

        shifts = self.library_indexes.first_rgroup_by_reaction

        asynthon2product = []
        for i, (reaction_id, synthon_ids) in enumerate(keys):
            for rgroup_id, synthon_id in enumerate(synthon_ids):
                key = (synthon_id, rgroup_id + shifts[reaction_id].item())
                asynthon2product.append([self.asynthon_mapper[key], i])

        asynthon2product = torch.tensor(asynthon2product, device=device).T

        pred_embeds = self.factorizer.embed_predictor_pooling_function(
            synthon_associative_embeds,
            asynthon2product,
        )

        return pred_embeds

    def calculate_synthon_associative_embeds(
        self,
        library_tensors: dict[str, Tensor],
        library_indexes: dict[str, LongTensor],
        batch_size: int = 1024,
    ):
        device = next(self.factorizer.parameters()).device
        synthon_feats = library_tensors["synthon_feats"].to(device)
        rgroup_feats = library_tensors["rgroup_feats"].to(device)
        reaction_feats = library_tensors["reaction_feats"].to(device)
        rgroup2reaction = library_indexes["rgroup2reaction"].to(device)
        synthon2rgroup = library_indexes["synthon2rgroup"].to(device)

        # (n_rgroups, embed_dim, associative_dim)
        rgroup_associative_keys = (
            self.factorizer.rgroup_associative_key_encoder(
                rgroup_feats[rgroup2reaction[0]]
                + reaction_feats[rgroup2reaction[1]]
            ).view(-1, self.embed_dim, self.associative_dim)
        )

        # (n_synthons, associative_dim)
        synthon_associative_values = (
            self.factorizer.synthon_associative_value_encoder(
                synthon_feats
            ).view(-1, self.associative_dim, 1)
        )

        # (O(n_rgroups * n_synthons), embed_dim)
        synthon_associative_embeds = []
        (start_idx, end_idx) = (0, batch_size)
        while start_idx < synthon2rgroup.size(1):
            row = synthon2rgroup[1, start_idx:end_idx]
            col = synthon2rgroup[0, start_idx:end_idx]
            val = torch.bmm(
                rgroup_associative_keys[row],
                synthon_associative_values[col],
            ).squeeze(-1)
            synthon_associative_embeds.append(val)
            (start_idx, end_idx) = (end_idx, end_idx + batch_size)

        return torch.cat(synthon_associative_embeds, 0)

    def calculate_synthon_associative_contribs(
        self,
        library_tensors,
        library_indexes,
    ):
        scaled_weight = self.probe.weight.T / self.probe.output_scale
        synthon_associative_embeds = library_tensors[
            "synthon_associative_embeds"
        ]
        synthon_associative_contribs = (
            synthon_associative_embeds @ scaled_weight
        )
        return synthon_associative_contribs

    def build_apex_libtree(self) -> list[list[list[int]]]:
        shifts = self.library_indexes.first_rgroup_by_reaction
        library_tree = []
        for reaction_id, rgroup_synthon_ids in enumerate(self.dataset.libtree):
            library_tree.append([])
            for rgroup_id, synthon_ids in enumerate(rgroup_synthon_ids):
                library_tree[reaction_id].append([])
                for synthon_id in synthon_ids:
                    key = (synthon_id, rgroup_id + shifts[reaction_id].item())
                    idx = self.asynthon_mapper[key]
                    library_tree[reaction_id][rgroup_id].append(idx)
        return library_tree

    @torch.no_grad()
    def run_apex_search(
        self,
        k: int,
        objective: str,
        constraints: Optional[
            dict[str, tuple[Optional[float], Optional[float]]]
        ] = None,
        probe: Optional[LinearProbe] = None,
        maximize: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> pd.DataFrame:
        """
        This is the primary method in APEX, used to conduct a search of the
        associated CSLDataset with the goal of finding the `k` most optimal
        compounds based on the provided `linear` predictors according to the
        `objective` and subject to `constraints`. It uses a similar log reward
        definition as in NGT. To codify the objective as a constraint, a large
        lower bound (cf. upper bound if `maximize` is False) of `alpha` (cf.
        negative `alpha`) is set on the (normalized) objective with a scale of
        `beta`. Under certain conditions (namely that `beta` >> `alpha` are both
        large), selecting the top `k` based on the log reward defined in this
        manner is equivalent to lexicographical sorting first on the log reward
        defined on just the `constraints` and second on the `objective` value.
        """
        # Setup
        device = next(self.parameters()).device
        self.to(device)
        default_probe = False
        if probe is None:
            default_probe = True
            assert isinstance(self.probe, LinearProbe)
            probe = self.probe
        probe = probe.to(device)

        logger.info(
            f"Running APEX screen of CSL with {len(self.dataset):,} compounds. "
            f"Will return the top {k:,} compounds."
        )
        times = [time.time()]

        # Process constraints
        constraints = dict({} if constraints is None else constraints)
        endpt_indexes = []
        if len(constraints) > 0:
            constr_lower = []
            constr_upper = []
            for i, (name, (lb, ub)) in enumerate(constraints.items()):
                assert name in probe.output_names
                constr_idx = probe.name_to_idx[name]
                lb = float("-inf" if lb is None else lb)
                ub = float("+inf" if ub is None else ub)
                assert (
                    lb <= ub
                ), "Lower bound must not be larger than upper bound."
                bias = probe.bias[constr_idx].item()
                scale = probe.output_scale[constr_idx].item()
                constr_lower.append((lb - bias) / scale)
                constr_upper.append((ub - bias) / scale)
                endpt_indexes.append(constr_idx)

            # Get weights, bounds, and indexes as tensors
            constr_bounds = torch.tensor(
                [constr_lower, constr_upper],
                device=device,
            )
            constr_indexes = (
                torch.arange(len(constraints), device=device)
                .view(1, -1)
                .repeat(2, 1)
            )
        else:
            constr_indexes = constr_bounds = None

        # Get objective weights and bias
        obj_sign = 2 * maximize - 1
        obj_idx = probe.name_to_idx[objective]
        obj_bias = obj_sign * probe.bias[obj_idx].item()
        obj_index = torch.tensor([len(constraints)], device=device)
        endpt_indexes.append(obj_idx)

        # Combine constraint and objective weights
        if default_probe:
            synthon_associative_contribs = (
                self.library_tensors.synthon_associative_contribs[
                    :, endpt_indexes
                ]
            )
        else:
            scaled_weight = (
                probe.weight[endpt_indexes].T
                / probe.output_scale[endpt_indexes]
            )
            synthon_associative_contribs = (
                self.library_tensors.synthon_associative_embeds @ scaled_weight
            )
        synthon_associative_contribs[:, -1] *= (
            obj_sign * probe.output_scale[obj_idx].item()
        )

        # Get topk predictions
        (apex_values, indexes) = topk(
            contributions=synthon_associative_contribs,
            library_tree=self.apex_libtree,
            topk=k,
            objective_indices=obj_index,
            constraint_indices=constr_indexes,
            constraint_bounds=constr_bounds,
            default_chunk_size=chunk_size,
        )
        (apex_constr_values, apex_obj_values) = apex_values.T
        apex_obj_values.add_(obj_bias)
        apex_obj_values.mul_(obj_sign)
        apex_constr_values = apex_constr_values.tolist()
        apex_obj_values = apex_obj_values.tolist()

        # Convert indexes to reaction and synthon IDs
        top_reaction_ids = indexes[:, 0].tolist()
        top_contrib_ids = indexes[:, 1:]
        top_synthon_ids = []
        for row in top_contrib_ids.T:
            idx = self.library_indexes.synthon2rgroup[0][row]
            idx[row == -1] = -1
            top_synthon_ids.append(idx)

        top_synthon_ids = torch.stack(top_synthon_ids, -1).tolist()
        top_synthon_ids = [
            [n for n in sublist if n >= 0] for sublist in top_synthon_ids
        ]

        times += [time.time()]
        time_elapsed = times[-1] - times[-2]
        logger.info(
            f"Screening complete in {time_elapsed:.2f} seconds. Converting top "
            f"{k:,} compounds to SMILES."
        )

        # Get compound names and convert compounds to SMILES
        name_list = []
        smiles_list = []
        apex_obj_value_list = []
        apex_constr_value_list = []
        zipper = zip(top_reaction_ids, top_synthon_ids)
        for i, (reaction_id, synthon_ids) in enumerate(zipper):
            name_list.append(self.dataset.key2name(reaction_id, synthon_ids))
            smiles_list.append(
                self.dataset.key2smiles(reaction_id, synthon_ids)
            )
            apex_obj_value_list.append(apex_obj_values[i])
            apex_constr_value_list.append(apex_constr_values[i])

        times += [time.time()]
        time_elapsed = times[-1] - times[-2]
        logger.info(
            f"All {k:,} products have been converted to SMILES in "
            f"{time_elapsed:.2f} seconds."
        )

        # Construct output dataframe
        df = pd.DataFrame(
            {
                "name": name_list,
                "smiles": smiles_list,
                "apex_obj_value": apex_obj_value_list,
                "apex_constr_value": apex_constr_value_list,
            }
        )

        time_elapsed = time.time() - times[0]
        logger.info(
            f"Returning dataframe with top {k:,} products. Total elapsed "
            f"time: {time_elapsed:.2f} seconds."
        )

        return df

    def encode_library(
        self,
        synthon_feats: Tensor,
        library_indexes: dict[str, LongTensor],
        calculate_contributions: bool = False,
    ) -> dict[str, Tensor]:
        # Move synthon features to same device as CSLVAE
        device = next(self.parameters()).device
        synthon_feats = synthon_feats.to(device)

        # Get library indexes
        synthon2rgroup = library_indexes["synthon2rgroup"]
        rgroup2reaction = library_indexes["rgroup2reaction"]

        # Encode library
        rgroup_feats = self.factorizer.rgroup_encoder_p1(
            self.factorizer.rgroup_encoder_pooling_function(
                self.factorizer.rgroup_encoder_p0(synthon_feats), synthon2rgroup
            )
        )
        reaction_feats = self.factorizer.reaction_encoder_p1(
            self.factorizer.reaction_encoder_pooling_function(
                self.factorizer.reaction_encoder_p0(rgroup_feats),
                rgroup2reaction,
            )
        )
        library_tensors = {
            "synthon_feats": synthon_feats,
            "rgroup_feats": rgroup_feats,
            "reaction_feats": reaction_feats,
        }
        library_tensors["synthon_associative_embeds"] = (
            self.calculate_synthon_associative_embeds(
                library_tensors,
                library_indexes,
            )
        )
        if calculate_contributions:
            library_tensors["synthon_associative_contribs"] = (
                self.calculate_synthon_associative_contribs(
                    library_tensors,
                    library_indexes,
                )
            )
        return library_tensors

    @torch.no_grad()
    def update_synthon_embeddings(
        self,
        batch_size: int = 100,
        logging_iterations: int = 100,
    ) -> None:
        """
        This method pre-computes the embeddings for all synthons from the
        associated CSLDataset and accordingly updates the nn.ParameterDict
        `library_tensors`.

        Args:
            batch_size: The batch size specified for the dataloader in
            calculating the synthon embeddings.
            logging_iterations: How often to log progress.
        """
        self.eval()
        device = next(self.factorizer.parameters()).device

        smiles = self.dataset.synthon_smiles
        embeds = []
        dataset = LabeledSmilesDataset(pd.DataFrame({"smiles": smiles}))
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        max_iterations = math.ceil(len(dataset) / batch_size)
        logger.info(
            f"Computing embeddings for {len(dataset):,} synthons over "
            f"{max_iterations:,} iterations."
        )
        for batch_iter, batch in enumerate(dataloader):
            synthons = batch.get("mols").to(device)
            embeds.append(self.factorizer.synthon_graph_encoder(synthons))
            if batch_iter % logging_iterations == 0:
                logger.info(
                    f"Completed iteration {batch_iter}/{max_iterations}."
                )
        del dataloader, dataset
        embeds = torch.cat(embeds, 0)
        self.library_tensors["synthon_feats"][:] = embeds.to(device)
        logger.info("All synthon embeddings have been computed.")

    @torch.no_grad()
    def update_library_tensors(
        self,
        update_synthon_embeddings: bool = True,
        **kwargs,
    ) -> None:
        """
        This method pre-computes all tensors required for decoding queries into
        the associated CSLDataset and accordingly updates the nn.ParameterDict
        `full_library_tensors`.

        Args:
            update_synthon_embeddings: Whether to update the synthon
              embeddings. If True, this will run the `update_synthon_embeddings`
              method. Setting to False if you have already pre-computed these
              embeddings will let this method run faster.
        """
        start_time = time.time()
        self.eval()
        device = next(self.parameters()).device

        logger.info("Updating library tensors.")

        if update_synthon_embeddings:
            self.update_synthon_embeddings(**kwargs)

        library_tensors = self.encode_library(
            synthon_feats=self.library_tensors["synthon_feats"],
            library_indexes=self.library_indexes,
            calculate_contributions=(self.probe is not None),
        )
        self.library_tensors["rgroup_feats"][:] = library_tensors[
            "rgroup_feats"
        ].to(device)
        self.library_tensors["reaction_feats"][:] = library_tensors[
            "reaction_feats"
        ].to(device)
        self.library_tensors["synthon_associative_embeds"][:] = library_tensors[
            "synthon_associative_embeds"
        ].to(device)
        if self.probe is not None:
            self.library_tensors["synthon_associative_contribs"][:] = (
                library_tensors["synthon_associative_contribs"].to(device)
            )
        del library_tensors
        end_time = time.time()
        mins_elapsed = (end_time - start_time) / 60.0
        logger.info(
            f"All library tensors have been updated. Elapsed time: "
            f"{mins_elapsed:.2f}."
        )
