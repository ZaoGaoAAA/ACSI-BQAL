"""Core ACSI and BQAL modules described in the accompanying paper.

Tensor conventions follow OpenTAD: temporal features use ``[B, C, T]``.
Boundary coordinates passed to :class:`BoundaryQualityHead` are expressed in
feature-grid units, with valid locations ranging from 0 to ``T - 1``.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor, nn


class ChannelLayerNorm(nn.Module):
    """Layer normalization for channel-first temporal tensors."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalBranch(nn.Module):
    """Depthwise temporal convolution followed by pointwise projection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, 1, bias=False),
            ChannelLayerNorm(channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class AdaptiveCrossScaleInteraction(nn.Module):
    """Adaptive Cross-Scale Interaction (ACSI).

    Multiple receptive-field branches are compared at every temporal location.
    Attention is computed across branches rather than across time, so the added
    cost remains linear in sequence length for a fixed number of branches.
    """

    def __init__(
        self,
        channels: int,
        interaction_dim: int = 128,
        branch_specs: Iterable[Tuple[int, int]] = ((3, 1), (3, 2), (3, 4)),
    ) -> None:
        super().__init__()
        specs = tuple(branch_specs)
        if len(specs) < 2:
            raise ValueError("ACSI requires at least two receptive-field branches")
        self.branches = nn.ModuleList(
            TemporalBranch(channels, kernel_size, dilation)
            for kernel_size, dilation in specs
        )
        self.query = nn.Linear(channels, interaction_dim, bias=False)
        self.key = nn.Linear(channels, interaction_dim, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.scale_weight = nn.Linear(channels, 1)
        self.output = nn.Linear(channels, channels, bias=False)
        self.scale = interaction_dim**-0.5

    def forward(self, x: Tensor, return_weights: bool = False):
        if x.ndim != 3:
            raise ValueError(f"expected [B, C, T], received {tuple(x.shape)}")

        # [B, T, S, C]
        branches = torch.stack(
            [branch(x).transpose(1, 2) for branch in self.branches], dim=2
        )
        attention = torch.softmax(
            torch.matmul(self.query(branches), self.key(branches).transpose(-1, -2))
            * self.scale,
            dim=-1,
        )
        interacted = branches + torch.matmul(attention, self.value(branches))
        weights = torch.softmax(self.scale_weight(interacted).squeeze(-1), dim=-1)
        fused = (interacted * weights.unsqueeze(-1)).sum(dim=2)
        enhanced = x.transpose(1, 2) + self.output(fused)
        enhanced = enhanced.transpose(1, 2)
        if return_weights:
            return enhanced, weights, attention
        return enhanced


def _linear_sample_1d(features: Tensor, coordinates: Tensor) -> Tensor:
    """Differentiably sample ``[B, C, T]`` features at ``[B, N, M]`` points."""

    if features.ndim != 3 or coordinates.ndim != 3:
        raise ValueError("features must be [B, C, T] and coordinates [B, N, M]")
    batch, channels, length = features.shape
    if coordinates.shape[0] != batch:
        raise ValueError("batch dimensions do not match")

    coordinates = coordinates.clamp(0, length - 1)
    left = coordinates.floor().long()
    right = (left + 1).clamp(max=length - 1)
    right_weight = coordinates - left.to(coordinates.dtype)
    left_weight = 1.0 - right_weight

    expanded = features.unsqueeze(2).expand(-1, -1, coordinates.shape[1], -1)
    left_value = torch.gather(expanded, 3, left.unsqueeze(1).expand(-1, channels, -1, -1))
    right_value = torch.gather(expanded, 3, right.unsqueeze(1).expand(-1, channels, -1, -1))
    sampled = left_value * left_weight.unsqueeze(1) + right_value * right_weight.unsqueeze(1)
    return sampled.permute(0, 2, 3, 1)  # [B, N, M, C]


class BoundaryQualityHead(nn.Module):
    """Estimate separate starting- and ending-boundary reliability."""

    def __init__(self, channels: int, hidden_channels: int = 128, radius: int = 2) -> None:
        super().__init__()
        if radius < 0:
            raise ValueError("radius must be non-negative")
        self.radius = radius
        self.register_buffer(
            "offsets", torch.arange(-radius, radius + 1, dtype=torch.float32), persistent=False
        )

        def quality_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(2 * channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 1),
            )

        self.start_head = quality_head()
        self.end_head = quality_head()

    def _boundary_feature(self, features: Tensor, positions: Tensor) -> Tensor:
        coordinates = positions.unsqueeze(-1) + self.offsets.to(positions)
        return _linear_sample_1d(features, coordinates).mean(dim=2)

    def forward(
        self,
        features: Tensor,
        center_features: Tensor,
        start_positions: Tensor,
        end_positions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if center_features.ndim != 3:
            raise ValueError("center_features must use [B, N, C]")
        start_feature = self._boundary_feature(features, start_positions)
        end_feature = self._boundary_feature(features, end_positions)
        start_quality = torch.sigmoid(
            self.start_head(torch.cat((center_features, start_feature), dim=-1)).squeeze(-1)
        )
        end_quality = torch.sigmoid(
            self.end_head(torch.cat((center_features, end_feature), dim=-1)).squeeze(-1)
        )
        return start_quality, end_quality


def boundary_quality_targets(
    predicted_start: Tensor,
    predicted_end: Tensor,
    target_start: Tensor,
    target_end: Tensor,
    kappa: float = 4.0,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Construct duration-normalized soft targets with detached predictions."""

    duration = (target_end - target_start).clamp_min(eps)
    start_error = (predicted_start.detach() - target_start).abs() / duration
    end_error = (predicted_end.detach() - target_end).abs() / duration
    return torch.exp(-kappa * start_error), torch.exp(-kappa * end_error)


def calibrate_scores(class_scores: Tensor, start_quality: Tensor, end_quality: Tensor) -> Tensor:
    """Apply geometric boundary-quality calibration to class probabilities."""

    quality = torch.sqrt((start_quality * end_quality).clamp_min(0.0))
    return class_scores * quality.unsqueeze(-1)
