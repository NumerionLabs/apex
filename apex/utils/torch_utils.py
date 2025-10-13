# Standard
import copy

# Third party
import torch

# APEX
from apex.utils.other_utils import flatten_list


def convert_1d_to_2d_indexes(indexes: torch.LongTensor) -> torch.LongTensor:
    assert indexes.ndim == 1
    return torch.stack(
        tensors=[
            torch.arange(
                indexes.size(0), dtype=torch.long, device=indexes.device,
            ),
            indexes,
        ],
        dim=0,
    )


def build_library_indexes(
    libtree: list[list[list[int]]], device: str = "cpu"
) -> dict[str, torch.LongTensor]:
    # Prepare synthons
    orig_synthons: List[int] = sorted(set(flatten_list(flatten_list(libtree))))

    synthons_mapper = {k: i for i, k in enumerate(orig_synthons)}
    synthons_mapper_reverse = {i: k for k, i in synthons_mapper.items()}

    # Renumber the synthon IDs
    libtree = copy.deepcopy(libtree)
    for i, ix in enumerate(libtree):
        for j, jx in enumerate(ix):
            libtree[i][j] = list(map(lambda k: synthons_mapper[k], jx))

    # Form all mapping indexes
    libtree_1l: List[List[int]] = flatten_list(libtree)
    libtree_2l: List[int] = flatten_list(libtree_1l)
    synthons: List[int] = sorted(set(libtree_2l))

    orig_synthon_ids = torch.tensor(
        [synthons_mapper_reverse[i] for i in synthons], dtype=torch.long
    )
    n_reactions = torch.tensor(len(libtree), dtype=torch.long)
    n_rgroups = torch.tensor(len(libtree_1l), dtype=torch.long)
    n_synthons = torch.tensor(len(synthons), dtype=torch.long)

    synthon2rgroup_by_reaction = [
        list(map(lambda x: (x, i), idx)) for i, idx in enumerate(libtree_1l)
    ]

    n_rgroups_by_reaction = torch.tensor(
        [len(idx) for idx in libtree], dtype=torch.long
    )
    n_synthons_by_rgroup = torch.tensor(
        [len(idx) for idx in libtree_1l], dtype=torch.long
    )

    first_rgroup_by_reaction = torch.nn.functional.pad(
        n_rgroups_by_reaction, (1, 0)
    ).cumsum(0)[:-1]
    synthon2rgroup = torch_lexsort(
        torch.tensor(
            flatten_list(synthon2rgroup_by_reaction), dtype=torch.long
        ).T
    )

    rgroup2reaction = convert_1d_to_2d_indexes(
        torch.arange(n_reactions, dtype=torch.long).repeat_interleave(
            n_rgroups_by_reaction
        )
    )

    synthons_by_rgroup = [
        torch.tensor(idx, dtype=torch.long) for idx in libtree_1l
    ]

    library_indexes = {
        "n_reactions": n_reactions,
        "n_rgroups": n_rgroups,
        "n_synthons": n_synthons,
        "orig_synthon_ids": orig_synthon_ids,
        "synthon2rgroup": synthon2rgroup,
        "rgroup2reaction": rgroup2reaction,
        "n_rgroups_by_reaction": n_rgroups_by_reaction,
        "n_synthons_by_rgroup": n_synthons_by_rgroup,
        "first_rgroup_by_reaction": first_rgroup_by_reaction,
        **{
            f"synthons_where_rgroup_{i}": idx
            for i, idx in enumerate(synthons_by_rgroup)
        },
    }
    return {k: v.to(device) for k, v in library_indexes.items()}
