# Third party
import torch
from torch import nn, Tensor

# Atomwise
from apex.data.atom_bond_features import Vocabulary
from apex.data.torch_graph import TorchGraph
from apex.nn.mpnn import EdgeMessagePassingNetwork


class LigandEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mpnn_params: dict,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.mpnn_params = dict(mpnn_params)
        self.atom_embedder = nn.Linear(
            in_features=Vocabulary.get_num_node_features(),
            out_features=self.mpnn_params.get("node_dim"),
        )
        self.bond_embedder = nn.Linear(
            in_features=Vocabulary.get_num_edge_features(),
            out_features=self.mpnn_params.get("edge_dim"),
        )
        self.molecule_encoder = EdgeMessagePassingNetwork(**mpnn_params)
        self.molecule_linear = nn.Linear(
            in_features=self.mpnn_params.get("graph_dim"),
            out_features=self.embed_dim,
            bias=False,
        )

    def forward(self, molecules: TorchGraph) -> Tensor:
        device = next(self.parameters()).device
        molecules = molecules.to(device)
        (atom_feats, bond_feats, molecule_feats) = self.molecule_encoder(
            node_feats=self.atom_embedder(molecules.node_features),
            edge_feats=self.bond_embedder(molecules.edge_features),
            edge_index=molecules.edge_index,
            graph_feats=torch.zeros(
                (len(molecules), self.mpnn_params.get("graph_dim")),
                device=device,
                dtype=torch.float,
            ),
            graph_index=molecules.graph_index,
        )
        return self.molecule_linear(molecule_feats)

    @classmethod
    def load(cls, path: str) -> "LigandEncoder":
        state_dict = torch.load(path, weights_only=False)
        embed_dim = state_dict["embed_dim"]
        mpnn_params = state_dict["mpnn_params"]
        model = cls(embed_dim, mpnn_params)
        model.load_state_dict(state_dict["model_state_dict"])
        return model

    def save(self, path: str):
        state_dict = {
            "model_state_dict": self.state_dict(),
            "embed_dim": self.embed_dim,
            "mpnn_params": self.mpnn_params,
        }
        torch.save(state_dict, path)
