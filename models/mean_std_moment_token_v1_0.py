"""Lightweight Mean--Std moment token for point-set features.

This module intentionally contains no graph construction, message passing,
classification head, or category-label-dependent operation.  It reads a
pre-pooling feature set with shape ``[B, D, N]`` and returns a memory-query
token with shape ``[B, D, 1]``.
"""

import torch
import torch.nn as nn


class MeanStdMomentTokenV1(nn.Module):
    """Aggregate all point features using first- and second-order moments.

    The token is

        LayerNorm(a * mean(H) + b * std(H)),

    where ``a`` and ``b`` are learnable channel-wise scales.  Population
    variance is computed explicitly in FP32 so that ``N == 1`` remains finite
    and mixed-precision training does not underflow the variance epsilon.
    """

    def __init__(
        self,
        feature_dim=256,
        eps=1.0e-5,
        layer_norm_affine=False,
        mean_scale_init=1.0,
        std_scale_init=1.0,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.feature_dim = int(feature_dim)
        self.eps = float(eps)
        self.mean_scale = nn.Parameter(
            torch.full((self.feature_dim,), float(mean_scale_init))
        )
        self.std_scale = nn.Parameter(
            torch.full((self.feature_dim,), float(std_scale_init))
        )
        self.layer_norm = nn.LayerNorm(
            self.feature_dim,
            elementwise_affine=bool(layer_norm_affine),
        )

    def forward(self, features):
        """Return a ``[B, D, 1]`` token from ``[B, D, N]`` features."""
        if features.ndim != 3:
            raise ValueError(
                "MeanStdMomentTokenV1 expects [B, D, N], "
                f"got shape {tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected D={self.feature_dim}, got D={features.shape[1]}"
            )
        if features.shape[2] < 1:
            raise ValueError("MeanStdMomentTokenV1 requires at least one point")

        # Compute population moments in FP32 for AMP stability.  The explicit
        # mean-square expression avoids torch.std's sample-variance NaN at N=1.
        stats = features.float()
        mean = stats.mean(dim=2)
        centered = stats - mean.unsqueeze(2)
        variance = centered.square().mean(dim=2)
        std = torch.sqrt(variance + self.eps)

        moment = (
            self.mean_scale.float() * mean
            + self.std_scale.float() * std
        )
        moment = moment.to(dtype=features.dtype)
        token = self.layer_norm(moment)
        return token.unsqueeze(2)
