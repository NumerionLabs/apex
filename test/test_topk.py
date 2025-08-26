# Standard
import itertools
import random

# Third party
import pytest
import torch
import torch.nn.functional as F

# Atomwise
import apex


def libtree_factory():
    (min_synth, max_synth) = (5, 10)
    (min_rxn, max_rxn) = (10, 20)

    size_tree = [
        [
            random.randint(min_synth, max_synth)
            for _ in range(random.randint(2, 3))
        ]
        for _ in range(random.randint(min_rxn, max_rxn))
    ]

    size = sum(sum(rxn) for rxn in size_tree)

    tree = []
    offset = 0
    for group_sizes in size_tree:
        groups = []
        tree.append(groups)
        for group_size in group_sizes:
            ids = list(range(offset, offset + group_size))
            offset += group_size
            groups.append(ids)

    return (tree, size)


def ref_topk(
    contribs,
    libtree,
    topk,
    descending,
    quad_ids=None,
    obj_ids=None,
    const_ids=None,
    const_bounds=None,
):
    if contribs.ndim == 1:
        contribs = contribs.unsqueeze(dim=1)

    if const_ids is not None:
        (dest_ids, src_ids) = const_ids.unbind(dim=0)
        (lower, upper) = torch.zeros(2, contribs.size(dim=1)).unbind(dim=0)
        lower[dest_ids] = const_bounds[0]
        upper[dest_ids] = const_bounds[1]

    results = []
    for reaction_id, reaction in enumerate(libtree):
        synthon_ids = torch.tensor(list(itertools.product(*reaction)))
        synthon_contribs = contribs[synthon_ids].sum(dim=1)

        if quad_ids is not None:
            squared_contribs = synthon_contribs[:, quad_ids].square()
            synthon_contribs[:, quad_ids] = squared_contribs

        if const_ids is not None:
            synthon_consts_src = synthon_contribs[:, src_ids]
            synthon_consts = torch.zeros_like(synthon_contribs)
            synthon_consts.scatter_add_(
                dim=1,
                src=synthon_consts_src,
                index=dest_ids.expand_as(synthon_consts_src),
            )
            synthon_slacks = -(
                (lower - synthon_consts).relu().sum(dim=1) +
                (synthon_consts - upper).relu().sum(dim=1)
            )
        else:
            synthon_slacks = torch.zeros(synthon_ids.size(dim=0))

        if obj_ids is not None:
            synthon_objs = synthon_contribs[:, obj_ids].sum(dim=1)
        else:
            synthon_objs = synthon_contribs.sum(dim=1)

        pad_size = 4 - synthon_ids.size(dim=1)
        padded_ids = F.pad(synthon_ids, (0, pad_size), value=-1)

        for ids, slack, obj in zip(
            padded_ids.tolist(),
            synthon_slacks.tolist(),
            synthon_objs.tolist(),
        ):
            results.append((reaction_id, *ids, slack, obj))

    results.sort(key=lambda result: result[5:], reverse=True)
    results = results[:topk]
    results.sort(key=lambda result: result[5:], reverse=descending)

    scores = torch.tensor([result[5:] for result in results])
    ids = torch.tensor([result[:5] for result in results])

    return (scores, ids)


@pytest.mark.parametrize("descending", (True, False))
@pytest.mark.parametrize("topk", (1, 2, 5))
@pytest.mark.parametrize("const", (False, True))
@pytest.mark.parametrize("obj", (False, True))
@pytest.mark.parametrize("quad", (False, True))
@pytest.mark.parametrize("trial", range(10))
@pytest.mark.parametrize("use_gpu", (False, True))
def test_topk(use_gpu, trial, quad, obj, const, topk, descending):
    """Check correctness of the CPU and CUDA top-k implementations."""

    def to_device(obj):
        return obj.cuda() if use_gpu and torch.is_tensor(obj) else obj

    # setup.
    (libtree, size) = libtree_factory()

    nterms = 1 * (4 if quad else 1) * (8 if const else 1)

    if quad:
        quad_ids = torch.tensor((0, 1))
    else:
        quad_ids = None

    if obj:
        obj_ids = torch.tensor((0,))
    else:
        obj_ids = None

    if const:
        const_ids = torch.tensor(((1, 1, 2, 2), (0, 1, 4, 5)))
        const_bounds = 2 * torch.rand(2, 2) - 1
        repeats = torch.tensor((2, 2))
        const_bounds = const_bounds.repeat_interleave(repeats, dim=1)
        const_bounds = const_bounds.sort(dim=0).values
    else:
        const_ids = None
        const_bounds = None

    contribs = torch.randn((size, nterms), dtype=torch.float)

    # test sanity of the return values.
    (scores, ids) = apex.topk(
        to_device(contribs),
        libtree,
        topk,
        descending=descending,
        quadratic_indices=to_device(quad_ids),
        objective_indices=to_device(obj_ids),
        constraint_indices=to_device(const_ids),
        constraint_bounds=to_device(const_bounds),
    )

    if use_gpu:
        scores = scores.cpu()
        ids = ids.cpu()

    assert torch.is_tensor(scores)
    assert torch.is_tensor(ids)

    assert scores.shape == (topk, 2)
    assert ids.shape == (topk, 5)

    assert ids.dtype == torch.int64
    assert scores.dtype == torch.float

    # test correctness of the return values.
    (ref_scores, ref_ids) = ref_topk(
        contribs,
        libtree,
        topk,
        descending,
        quad_ids,
        obj_ids,
        const_ids,
        const_bounds,
    )

    assert torch.allclose(scores, ref_scores, atol=1e-6)
    assert torch.equal(ids, ref_ids)
