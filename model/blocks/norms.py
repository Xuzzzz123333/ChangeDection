import torch.nn as nn


def resolve_group_count(channels: int, preferred_groups: int = 8) -> int:
    channels = int(channels)
    groups = max(1, min(int(preferred_groups), channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def group_norm(
    num_channels: int,
    preferred_groups: int = 8,
    affine: bool = True,
) -> nn.GroupNorm:
    return nn.GroupNorm(
        resolve_group_count(num_channels, preferred_groups),
        num_channels,
        affine=affine,
    )
