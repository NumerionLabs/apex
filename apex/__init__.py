# Standard
from pathlib import Path
from typing import Any, Optional

# Third party
import torch

torch.ops.load_library(Path(__file__).parent / "_ops.so")


def assert_same_device(*args: Any):
    """
    Check that all tensors reside on the same device.

    Args:
        arg: Any number of input arguments. Only the tensor arguments
            will be checked.

    Raises:
        RuntimeError: If the input tensors reside on multiple devices.
    """
    tensors = [arg for arg in args if torch.is_tensor(arg)]
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        names = ", ".join(repr(str(device)) for device in devices)
        raise RuntimeError(
            "expected tensors to be on the same device, "
            f"but found at least {len(devices)} devices: {names}"
        )


def topk(
    contributions: torch.Tensor,
    library_tree: list[list[list[int]]],
    topk: int = 1000,
    *,
    descending: bool = True,
    quadratic_indices: Optional[torch.Tensor] = None,
    objective_indices: Optional[torch.Tensor] = None,
    constraint_indices: Optional[torch.Tensor] = None,
    constraint_bounds: Optional[torch.Tensor] = None,
    default_chunk_size: int = 2**30,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Exhaustively search a library for the top :math:`k` molecules.

    Given a :attr:`library_tree` containing :math:`R` reactions, with each
    reaction :math:`r` containing :math:`C_r` components, and each component
    of that reaction having :math:`G_{r,c}` possible R-groups, this function
    accepts a one-dimensional tensor of R-group score :attr:`contributions`
    of size,

    .. math::
        n = \sum_{r=1}^R \sum_{c=1}^{C_r} G_{r,c}

    The score of a molecule :math:`m` has a score that is the sum of its
    R-group contributions,

    .. math::
        f(m) = \sum_{c=1}^{C_{r(m)}} s_{r(m),g_c(m)}

    where :math:`r(m)` denotes the reaction used to form molecule :math:`m`,
    :math:`g_c(m)` denotes the R-group used in position :math:`c` by molecule
    :math:`m`, and :math:`s_{r,g}` denotes the score contribution of that
    R-group.

    Alternatively, contributions may be passed in for :math:`T` different
    endpoints as a two-dimensional :math:`(n, 2T)` tensor, where the second
    dimension contains :math:`T` entries for lower-bounded contributions and
    :math:`T` entries for upper-bounded contributions. The resulting score of
    a molecule is then given by,

    .. math::
        \log R(m) = \sum_{\tau=1}^{2T} \max( f_\tau(m), 0 )

    Args:
        contributions: :math:`(n)` or :math:`(n, t)`, Objective contributions
            and pre-transformed constraint contributions for all R-groups.
        quadratic_indices: :math:`(n_q)`, Optional tensor indicating which
            second-dimension indices in :attr:`contributions` must be squared.
        objective_indices: :math:`(n_f)`, Optional tensor indicating which
            second-dimension indices in :attr:`contributions` are part of
            objective terms. If not provided, all indices will be treated
            as objective terms.
        constraint_indices: :math:`(2, n_c)`, Optional tensor indicating which
            second-dimension indices in :attr:`contributions` are part of
            constraint terms. The first row contains destination indices and
            the second row contains source indices, both of which are part
            of a *scatter-sum* operation.
        constraint_bounds: :math:`(2, n_c)`, Optional tensor containing the
            lower and upper bounds of each constraint term.
        library_tree: Nested list holding the combinatorial library structure.
        topk: :math:`k`, Maximum number of top-scoring molecules to return.
        descending: When true, return results in descending order. By default,
            results are sorted lexicographically, first by constraint violation
            and then by objective.
        default_chunk_size: Used to tune the chunking behavior of the GPU
            top-:math:`k` kernel.

    Returns:
        A :class:`tuple` containing

        - :math:`(k, 2)`, Scores of the top :math:`k` molecules, containing
          constraint slack values in in the first row and objective values
          in the second row.
        - :math:`(k, 5)`, Reaction and R-group indices of the top :math:`k`
          molecules.
    """
    assert_same_device(
        contributions,
        quadratic_indices,
        objective_indices,
        constraint_indices,
        constraint_bounds,
    )

    # sanity check the contributions tensor.
    if contributions.ndim == 1:
        contributions = contributions.unsqueeze(dim=1)

    if contributions.ndim != 2:
        raise ValueError("contributions must be one- or two-dimensional")

    if contributions.numel() == 0:
        raise ValueError("contributions must be non-empty")

    max_id = contributions.size(dim=1) - 1
    id_bounds = f"[0,{max_id}]"

    # sanity check the quadratic indices.
    if quadratic_indices is None:
        quadratic_indices = torch.empty(
            size=(0,),
            dtype=torch.long,
            device=contributions.device,
        )

    if quadratic_indices.ndim != 1:
        raise ValueError("quadratic indices must be one-dimensional")

    if quadratic_indices.lt(0).any() or quadratic_indices.gt(max_id).any():
        raise ValueError(f"quadratic indices are out of bounds {id_bounds}")

    # sanity check the objective indices.
    if objective_indices is None:
        objective_indices = torch.arange(
            start=0,
            end=max_id + 1,
            device=contributions.device,
        )

    if objective_indices.ndim != 1:
        raise ValueError("objective indices must be one-dimensional")

    if objective_indices.lt(0).any() or objective_indices.gt(max_id).any():
        raise ValueError(f"objective indices are out of bounds {id_bounds}")

    # sanity check the constraint indices.
    if constraint_indices is None:
        constraint_indices = torch.empty(
            size=(2, 0),
            dtype=torch.long,
            device=contributions.device,
        )

    if constraint_indices.ndim != 2 or constraint_indices.size(dim=0) != 2:
        raise ValueError("constraint indices must have shape (2,*)")

    if constraint_indices.lt(0).any() or constraint_indices.gt(max_id).any():
        raise ValueError(f"constraint indices are out of bounds {id_bounds}")

    # sanity check the constraint bounds.
    if constraint_bounds is None:
        constraint_bounds = torch.empty(
            size=(2, 0),
            dtype=contributions.dtype,
            device=contributions.device,
        )

    if constraint_bounds.shape != constraint_indices.shape:
        raise ValueError(
            f"constraint bounds shape: {tuple(constraint_bounds.shape)} "
            "does not match the "
            f"constraint indices shape: {tuple(constraint_indices.shape)}"
        )

    # run the topk operator.
    return torch.ops.apex.topk(
        contributions,
        quadratic_indices,
        objective_indices,
        constraint_indices,
        constraint_bounds,
        library_tree,
        topk,
        descending,
        default_chunk_size,
    )
