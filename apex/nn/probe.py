# Standard
from typing import Optional

# Third party
import torch
from torch import nn, Tensor


class LinearProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_names: list[str],
        epsilon: float = 1.0,
        momentum: float = 0.1,
        use_bias: bool = True,
    ):
        """
        Args:
          input_dim: Number of input dimensions.
          output_names: The names of the output dimensions, represented as a
            list of strings.
          epsilon: Ridge parameter in least squares estimation.
          momentum: Momentum term for updating buffers.
          use_bias: Whether to include a bias term.
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_names = list(map(str, output_names))
        self.output_dim = len(self.output_names)
        assert len(set(self.output_names)) == self.output_dim
        self.epsilon = float(epsilon)
        self.momentum = float(momentum)
        self.use_bias = bool(use_bias)
        self.register_buffer("output_bias", torch.zeros((self.output_dim,)))
        self.register_buffer("output_scale", torch.ones((self.output_dim,)))
        self.register_buffer(
            "weight", torch.ones((self.output_dim, self.input_dim))
        )
        self.register_buffer("bias", torch.zeros((self.output_dim,)))
        self.register_buffer("_n_train_iters", torch.tensor(-1))
        self.name_to_idx = {name: i for i, name in enumerate(self.output_names)}
        self.idx_to_name = {i: name for i, name in enumerate(self.output_names)}

    def forward(self, input: Tensor, values: Optional[Tensor] = None) -> Tensor:
        if values is None:
            return self.bias.view(1, -1) + input @ self.weight.T
        else:
            return self._leave_one_out_estimator(input, values)

    def _leave_one_out_estimator(self, input: Tensor, values: Tensor) -> Tensor:
        (x, y) = (input, values)
        device = x.device
        n, input_dim = x.shape
        ny, output_dim = y.shape
        assert n == ny

        idx = (
            torch.arange(n, device=device)
            .view(1, -1)
            .repeat(n, 1)[~torch.eye(n, dtype=torch.bool, device=device)]
        )

        x_loo = x[idx].view(n, n - 1, input_dim)
        y_loo = y[idx].view(n, n - 1, output_dim)

        output_bias_loo = y_loo.mean(1)
        output_scale_loo = y_loo.std(1, unbiased=True)
        input_bias_loo = x_loo.mean(1)
        input_scale_loo = x_loo.std(1, unbiased=True)

        if not self.use_bias:
            output_bias_loo = torch.zeros_like(output_bias_loo)
            output_scale_loo = torch.ones_like(output_scale_loo)
            input_bias_loo = torch.zeros_like(input_bias_loo)

        output_bias_loo_unsqueeze = output_bias_loo.unsqueeze(1)
        output_scale_loo_unsqueeze = output_scale_loo.unsqueeze(1)
        input_bias_loo_unsqueeze = input_bias_loo.unsqueeze(1)
        input_scale_loo_unsqueeze = input_scale_loo.unsqueeze(1)

        y_norm_loo = (
            y_loo - output_bias_loo_unsqueeze
        ) / output_scale_loo_unsqueeze
        x_norm_loo = (
            x_loo - input_bias_loo_unsqueeze
        ) / input_scale_loo_unsqueeze
        x_norm = (x - input_bias_loo) / input_scale_loo

        ridge = self.epsilon * torch.eye(input_dim, device=device).unsqueeze(0)
        beta_loo = torch.bmm(
            torch.linalg.inv(
                torch.bmm(x_norm_loo.transpose(1, 2), x_norm_loo) + ridge
            ),
            torch.bmm(x_norm_loo.transpose(1, 2), y_norm_loo),
        )
        yhat_norm = torch.bmm(x_norm.unsqueeze(1), beta_loo)
        yhat = (
            output_scale_loo_unsqueeze * yhat_norm + output_bias_loo_unsqueeze
        ).squeeze(1)

        if self.training:
            with torch.no_grad():
                output_bias_mean = output_bias_loo.mean(0, keepdims=True)
                output_scale_mean = output_scale_loo.mean(0, keepdims=True)
                input_bias_mean = input_bias_loo.mean(0, keepdims=True)
                input_scale_mean = input_scale_loo.mean(0, keepdims=True)
                beta_mean = beta_loo.mean(0)
                output_bias = output_bias_mean.view(-1)
                output_scale = output_scale_mean.view(-1)
                input_scaled_bias = input_bias_mean / input_scale_mean

                if self.use_bias:
                    weight = (
                        output_scale_mean.T / input_scale_mean * beta_mean.T
                    )
                    bias = output_bias_mean - output_scale_mean * (
                        input_scaled_bias @ beta_mean
                    )
                    bias = bias.view(-1)
                else:
                    weight = beta_mean.T
                    bias = torch.zeros_like(output_bias)

                if self._n_train_iters < 0:
                    self.output_bias.copy_(output_bias)
                    self.output_scale.copy_(output_scale)
                    self.bias.copy_(bias)
                    self.weight.copy_(weight)
                else:
                    delta = 1 - self.momentum
                    self.output_bias.add_(
                        delta * (output_bias - self.output_bias)
                    )
                    self.output_scale.add_(
                        delta * (output_scale - self.output_scale)
                    )
                    self.bias.add_((delta) * (bias - self.bias))
                    self.weight.add_(delta * (weight - self.weight))
                self._n_train_iters.add_(1)

        return yhat

    @classmethod
    def load(cls, path: str, **kwargs) -> "LinearProbe":
        state_dict = torch.load(path, **kwargs)
        input_dim = state_dict["input_dim"]
        output_names = state_dict["output_names"]
        epsilon = state_dict["epsilon"]
        momentum = state_dict["momentum"]
        use_bias = state_dict["use_bias"]
        model = cls(input_dim, output_names, epsilon, momentum, use_bias)
        model.load_state_dict(state_dict["model_state_dict"])
        return model

    def save(self, path: str):
        state_dict = {
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "output_names": self.output_names,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "use_bias": self.use_bias,
        }
        torch.save(state_dict, path)
