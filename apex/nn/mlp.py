# Standard
from typing import List, Optional

# Third party
from torch import nn, Tensor


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        normalization: bool = True,
        activation: type = nn.GELU,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = list(map(int, hidden_dims or []))
        self.output_dim = int(output_dim)
        self.normalization = bool(normalization)

        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if self.normalization:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation())
        self.mlp = nn.Sequential(
            *layers[: (-2 if self.normalization else -1)]
        )  # remove last activation/normalization

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class ResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_block_layers: int = 2,
        normalization: bool = True,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_block_layers = int(num_block_layers)
        self.normalization = bool(normalization)

        self.first_fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.last_fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.res_blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block_layers = []
            for _ in range(self.num_block_layers):
                if self.normalization:
                    block_layers.append(nn.LayerNorm(self.hidden_dim))
                block_layers.append(activation())
                block_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            # Set weights of last linear layer to 0. This makes the residual
            # update an identity operation at initialization
            block_layers[-1].weight.data.fill_(0)
            block = nn.Sequential(*block_layers)
            self.res_blocks.append(block)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_fc(x)
        for block in self.res_blocks:
            x = x + block(x)
        return self.last_fc(x)
