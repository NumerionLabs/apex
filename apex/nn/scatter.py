"""
Pure :mod:`torch` implementations of :mod:`torch_scatter` functions.

All functions here are built upon :meth:`torch.Tensor.scatter_reduce_`,
using the familiar function signatures of :mod:`torch_scatter`.
"""

# Standard
import sys
from typing import NamedTuple, Optional

# Third party
import torch
from torch import nn, Tensor


class ValuesAndIndices(NamedTuple):
    """
    Values and their associated indices from a source tensor.

    Args:
        values: Source tensor values.
        indices: Source tensor indices.
    """

    values: Tensor
    indices: Tensor


def add(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> Tensor:
    r"""
    Add values from a source tensor into destination indices.

    .. math::
        \mathrm{out}_i = \sum_j \mathrm{src}_j

    for all :math:`j` satisfying :math:`\mathrm{index}_j = i`.

    Args:
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        Scattered result.
    """
    return _scatter("sum", src, index, dim, dim_size)


def mean(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> Tensor:
    r"""
    Average values from a source tensor into destination indices.

    .. math::
        \mathrm{out}_i = \frac{1}{n_i} \sum_j \mathrm{src}_j

    for all :math:`j` satisfying :math:`\mathrm{index}_j = i`,
    where :math:`n_i` denotes the number of satisfying indices,

    .. math::
        n_i = \sum_j [\mathrm{index}_j = i]

    Args:
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        Scattered result.
    """
    index = _broadcast(index, src, dim)
    dim_size = _result_size(index, src, dim, dim_size)

    ones = index.new_ones(src.shape)
    sums = add(src, index, dim, dim_size)
    counts = add(ones, index, dim, dim_size).clamp_(min=1)

    return sums / counts


def amin(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> ValuesAndIndices:
    r"""
    Find minimum values from a source tensor using specified indices.

    .. math::
        \mathrm{out}_i = \min_j \mathrm{src}_j

    for all :math:`j` satisfying :math:`\mathrm{index}_j = i`.
    An index tensor is also returned which contains,

    .. math::
        \mathrm{outidx}_i = \arg\min_j \mathrm{src}_j

    Args:
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        A :class:`tuple` holding the scattered result and
        source tensor indices locating the resulting values.
    """
    index = _broadcast(index, src, dim)
    dim_size = _result_size(index, src, dim, dim_size)
    values = _scatter("amin", src, index, dim, dim_size)
    indices = _indices_from_values(values, src, index, dim, dim_size)

    return ValuesAndIndices(values, indices)


def amax(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> ValuesAndIndices:
    r"""
    Find maximum values from a source tensor using specified indices.

    .. math::
        \mathrm{out}_i = \max_j \mathrm{src}_j

    for all :math:`j` satisfying :math:`\mathrm{index}_j = i`.
    An index tensor is also returned which contains,

    .. math::
        \mathrm{outidx}_i = \arg\max_j \mathrm{src}_j

    Args:
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        A :class:`tuple` holding the scattered result and
        source tensor indices locating the resulting values.
    """
    index = _broadcast(index, src, dim)
    dim_size = _result_size(index, src, dim, dim_size)
    values = _scatter("amax", src, index, dim, dim_size)
    indices = _indices_from_values(values, src, index, dim, dim_size)

    return ValuesAndIndices(values, indices)


def softmax(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> Tensor:
    r"""
    Compute the softmax of values from a source tensor using destination
    indices.

    .. math::
        \mathrm{out}_i =
          \frac{ \exp(\mathrm{src}_i) }{ \sum_j \exp(\mathrm{src}_j) }

    for all :math:`j` satisfying :math:`\mathrm{index}_j = i`.

    Args:
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        Scattered result.
    """
    index = _broadcast(index, src, dim)
    dim_size = _result_size(index, src, dim, dim_size)

    max_per_index = amax(src, index, dim, dim_size).values
    max_per_src = max_per_index.gather(dim, index)

    shifted_src = src - max_per_src
    shifted_exp = shifted_src.exp()

    sum_per_index = add(shifted_exp, index, dim, dim_size)
    normalizing_const = sum_per_index.gather(dim, index)

    return shifted_exp / normalizing_const


def log_softmax(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
    eps: float = 1.0e-12,
) -> Tensor:
    r"""
    Compute the log-softmax of values from a source tensor using destination
    indices.

    .. math::
        \mathrm{out}_i = \mathrm{src}_i - \ln\sum_j \exp(\mathrm{src}_j)

    for all :math:`j` satisfying :math:`\mathrm{index}_j = i`.

    Args:
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        Scattered result.
    """
    index = _broadcast(index, src, dim)
    dim_size = _result_size(index, src, dim, dim_size)

    max_per_index = amax(src, index, dim, dim_size).values
    max_per_src = max_per_index.gather(dim, index)

    shifted_src = src - max_per_src
    shifted_exp = shifted_src.exp()

    sum_per_index = add(shifted_exp, index, dim, dim_size)
    log_normalizing_const = sum_per_index.add(eps).log().gather(dim, index)

    return shifted_src - log_normalizing_const


class Scatter(nn.Module):
    """
    Perform a scatter operation.

    Args:
        op: Operation to perform. Must be one of `add`, `mean`, `softmax`,
            or `log_softmax`.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.
        unbind_index: When true, only one- and two-dimensional index tensors
            will be supported, and two-dimensional index tensors will be
            unbound before scattering.

    Note:
        When :attr:`unbind_index` is true and a two-dimensional index tensor
        is passed as input, it must have a shape of :math:`(2, n)`. The first
        row will be used to index the source tensor, and the second row will
        be used in the scatter operation on that newly indexed source tensor.
    """

    def __init__(
        self,
        op: str = "add",
        dim: int = -1,
        dim_size: Optional[int] = None,
        unbind_index: bool = False,
    ):
        super().__init__()

        assert op in ("add", "mean", "softmax", "log_softmax")

        self.op = op
        self.dim = dim
        self.dim_size = dim_size
        self.unbind_index = unbind_index
        self.func = getattr(sys.modules[__name__], op)

    def extra_repr(self) -> str:
        """Return useful information about the module."""
        return (
            f"op={self.op}, dim={self.dim}, dim_size={self.dim_size}, "
            f"unbind_index={self.unbind_index}"
        )

    def forward(self, src: Tensor, index: Tensor) -> Tensor:
        """
        Scatter values from a source tensor using an index tensor.

        Args:
            src: Source tensor.
            index: Index tensor.

        Returns:
            Scattered result.
        """
        if self.unbind_index:
            assert index.ndim in (1, 2)
            if index.ndim == 2:
                (u, v) = index.unbind(dim=0)
                (src, index) = (src[u], v)

        return self.func(src, index, self.dim, self.dim_size)


def _broadcast(self: Tensor, other: Tensor, dim: int) -> Tensor:
    """
    Broadcast a tensor for use in a scatter function.

    Args:
        self: Input tensor to broadcast.
        other: Tensor that :attr:`self` will be broadcasted to match.
        dim: Dimension that will be indexed along in the scatter function.

    Returns:
        Input tensor expanded into the same shape as :attr:`other`.
    """
    if self.shape == other.shape:
        return self

    if dim < 0:
        dim = other.ndim + dim

    if self.dim() == 1:
        for _ in range(0, dim):
            self = self.unsqueeze(dim=0)

    for _ in range(self.dim(), other.dim()):
        self = self.unsqueeze(dim=-1)

    return self.expand_as(other)


def _result_size(
    index: Tensor,
    src: Tensor,
    dim: int,
    dim_size: Optional[int],
) -> int:
    """
    Determine the output dimension size of a scatter function.

    Args:
        index: Tensor of destination indices.
        src: Source tensor.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        Dimension size for the scatter result.
    """
    if dim_size is not None:
        return dim_size
    elif index.numel() == 0:
        return 0
    else:
        return index.max().item() + 1


def _scatter(
    reduce: str,
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> Tensor:
    """
    Wrapper around :meth:`torch.Tensor.scatter_reduce_`.

    Args:
        reduce: Reduction operation to apply.
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        Scattered result.
    """
    index = _broadcast(index, src, dim)

    shape = list(src.shape)
    shape[dim] = _result_size(index, src, dim, dim_size)

    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(dim, index, src, reduce, include_self=False)


def _indices_from_values(
    values: Tensor,
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_size: int,
) -> Tensor:
    """
    Determine the indices of scattered values produced by functions like
    :func:`amin` and :func:`amax`.

    Args:
        values: Scattered result.
        src: Source tensor.
        index: Tensor of destination indices.
        dim: Source dimension to index along.
        dim_size: Size of the output dimension.

    Returns:
        Indices of scattered elements. If no elements were scattered to a
        given output, then that output will be filled with the size of the
        scatter dimension :attr:`dim`.
    """
    indices = torch.arange(src.size(dim), device=src.device)
    indices = _broadcast(indices, src, dim)
    indices = values.gather(dim, index).eq(src).mul(indices)
    indices = _scatter("amax", indices, index, dim, dim_size)

    ones = index.new_ones(src.shape)
    noindex = add(ones, index, dim, dim_size).eq(0)
    indices.masked_fill_(noindex, src.size(dim))

    return indices
