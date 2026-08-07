"""PCN + label-free AdaMemory with a Mean--Std memory query.

Reviewer-control version with the following invariants:

* no GNN or graph construction;
* no classification head/loss and no category-label access;
* the Mean--Std token reads PCN's first-stage, pre-pooling features only;
* the token is used exclusively for memory addressing;
* PCN's original max-pooling/second-convolution/decoder path is unchanged.

The legacy memory-value aggregation is configurable so the tokenizer can first
be evaluated while holding the rest of the memory implementation fixed.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from extensions.chamfer_dist import ChamferDistanceL2
from models.circle_loss import CircleLoss

from .build import MODELS
from .mean_std_moment_token_v1_0 import MeanStdMomentTokenV1


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=True):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


def cos_similar(p, q, eps=1.0e-8):
    """Pairwise cosine similarity with finite zero-vector handling."""
    p = F.normalize(p, p=2, dim=-1, eps=eps)
    q = F.normalize(q, p=2, dim=-1, eps=eps)
    return p.matmul(q.transpose(-2, -1))


def convert_mask_to_similarity(
    similarity_matrix: Tensor,
    positive_pair_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Extract upper-triangular positive and negative pair similarities."""
    positive_matrix = positive_pair_mask.triu(diagonal=1)
    negative_matrix = positive_pair_mask.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.reshape(-1)
    positive_matrix = positive_matrix.reshape(-1)
    negative_matrix = negative_matrix.reshape(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


@MODELS.register_module()
class PCNAdaMemoryLabelFreeMeanStdQueryV1(nn.Module):
    """Label-free PCN AdaMemory using a moment-based memory query."""

    query_feature_dim = 256

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.number_fine = int(config.num_pred)
        self.encoder_channel = int(config.encoder_channel)
        self.grid_size = 4
        if self.number_fine % self.grid_size**2 != 0:
            raise ValueError("num_pred must be divisible by grid_size**2")
        self.number_coarse = self.number_fine // (self.grid_size**2)

        # Vanilla PCN backbone.  Keep construction order aligned with PCN.py.
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.query_feature_dim, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.number_coarse),
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        axis_a = torch.linspace(-0.05, 0.05, steps=self.grid_size).view(
            1, self.grid_size
        )
        axis_a = axis_a.expand(self.grid_size, self.grid_size).reshape(1, -1)
        axis_b = torch.linspace(-0.05, 0.05, steps=self.grid_size).view(
            self.grid_size, 1
        )
        axis_b = axis_b.expand(self.grid_size, self.grid_size).reshape(1, -1)
        folding_seed = torch.cat([axis_a, axis_b], dim=0).view(
            1, 2, self.grid_size**2
        )
        self.register_buffer("folding_seed", folding_seed)

        # Memory query dimensions are fixed to D=256 for this controlled study.
        self.encoder_embed_dim = int(config.encoder_config.embed_dim)
        self.decoder_embed_dim = int(config.decoder_config.embed_dim)
        if self.encoder_embed_dim != self.query_feature_dim:
            raise ValueError(
                "Mean--Std query study requires encoder_config.embed_dim=256; "
                f"got {self.encoder_embed_dim}"
            )

        # Construct the tokenizer after all vanilla PCN layers.  Its constant
        # initialization consumes no RNG and cannot perturb PCN initialization.
        self.query_readout = MeanStdMomentTokenV1(
            feature_dim=self.query_feature_dim,
            eps=float(config.get("moment_token_eps", 1.0e-5)),
            layer_norm_affine=bool(
                config.get("moment_token_layer_norm_affine", False)
            ),
            mean_scale_init=float(config.get("moment_token_mean_scale_init", 1.0)),
            std_scale_init=float(config.get("moment_token_std_scale_init", 1.0)),
        )

        self.memory_size = int(config.memory_size)
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        self.gating_alpha = nn.Parameter(torch.randn(1, self.memory_size))
        self.gating_beta = nn.Parameter(torch.randn(1, self.memory_size))
        self.memory_vector = nn.Parameter(
            torch.rand(
                self.memory_size,
                self.encoder_embed_dim + self.decoder_embed_dim,
            )
        )
        self.memory_query_mlp = MLP_CONV(
            in_channel=self.encoder_embed_dim,
            layer_dims=[384, self.encoder_embed_dim],
        )
        self.memory_key_mlp = MLP_CONV(
            in_channel=self.encoder_embed_dim,
            layer_dims=[384, self.encoder_embed_dim],
        )
        self.feature_align2 = MLP_CONV(
            in_channel=self.decoder_embed_dim,
            layer_dims=[512, 1024],
        )

        self.memory_value_aggregation = config.get(
            "memory_value_aggregation", "legacy_max"
        )
        if self.memory_value_aggregation not in {"legacy_max", "weighted_sum"}:
            raise ValueError(
                "memory_value_aggregation must be 'legacy_max' or "
                f"'weighted_sum', got {self.memory_value_aggregation!r}"
            )

        self.loss_func = ChamferDistanceL2()
        self.loss_circle_loss = CircleLoss(
            m=config.memory_circle_loss_m,
            gamma=config.memory_circle_loss_gamma,
        )

    def get_compactness_loss(self, input_dict):
        """Label-free grouping based only on nearest memory assignments."""
        memory_vector = input_dict["memory_vector"]
        partial_query = input_dict["partial_memory_query"].squeeze(2)
        complete_feature = input_dict["complete_memory_feature"].squeeze(2)
        lowest_num = int(self.config.get("not_similar_loss_num", 1))
        positive_thres = float(self.config.memory_circle_loss_m)
        negative_thres = 1.0 - positive_thres

        paired_feature = torch.cat((partial_query, complete_feature), dim=1)
        feature_memory_cos_sim = cos_similar(paired_feature, memory_vector)
        feature_feature_cos_sim = cos_similar(paired_feature, paired_feature)

        feature_memory_score, feature_memory_idx = feature_memory_cos_sim.max(-1)
        positive_pair_mask = (
            feature_memory_idx.unsqueeze(1) == feature_memory_idx.unsqueeze(0)
        )
        positive_sim, negative_sim = convert_mask_to_similarity(
            feature_feature_cos_sim,
            positive_pair_mask,
        )
        feature_circle_loss = self.loss_circle_loss(positive_sim, negative_sim)

        memory_positive_loss = F.relu(
            positive_thres - feature_memory_score
        ).mean()
        low_similarity, _ = feature_memory_score.sort()
        low_similarity = low_similarity[..., :lowest_num]
        memory_negative_loss = F.relu(low_similarity - negative_thres).mean()
        memory_feature_loss = memory_positive_loss + memory_negative_loss
        return feature_circle_loss, memory_feature_loss

    def get_sim_seperation_loss(self, input_dict):
        """Keep memory vectors separated; spelling retained for runner API."""
        negative_thres = 1.0 - float(self.config.memory_circle_loss_m)
        memory_vector = input_dict["memory_vector"]
        memory_similarity = cos_similar(memory_vector, memory_vector)
        identity = torch.eye(
            memory_similarity.shape[0],
            device=memory_similarity.device,
            dtype=memory_similarity.dtype,
        )
        memory_similarity = memory_similarity - identity
        return F.relu(memory_similarity - negative_thres).mean() / 2.0

    def get_rebuild_loss(self, input_dict, epoch=0):
        del epoch
        coarse_points = input_dict["coarse_points"]
        rebuild_points = input_dict["rebuild_points"]
        gt_points = input_dict["gt_points"]
        loss_coarse = self.loss_func(coarse_points, gt_points)
        loss_fine = self.loss_func(rebuild_points, gt_points)
        return gt_points.new_zeros(()), loss_coarse, loss_fine

    def _encode_partial(self, xyz):
        """Return original PCN latent and an independent memory query."""
        point_features = self.first_conv(xyz.transpose(2, 1))

        # Read-only pre-pooling branch.  This token never enters second_conv.
        memory_query = self.query_readout(point_features)

        # Original PCN path, deliberately kept identical to vanilla PCN.
        first_global = torch.max(point_features, dim=2, keepdim=True)[0]
        second_input = torch.cat(
            [first_global.expand(-1, -1, xyz.shape[1]), point_features],
            dim=1,
        )
        second_features = self.second_conv(second_input)
        pcn_latent = torch.max(second_features, dim=2, keepdim=False)[0]
        return pcn_latent, memory_query

    def _retrieve_memory(self, memory_query):
        keys = self.memory_vector[:, : self.encoder_embed_dim]
        values = self.memory_vector[:, -self.decoder_embed_dim :]

        query = self.memory_query_mlp(memory_query).squeeze(2)
        keys_projected = self.memory_key_mlp(keys.unsqueeze(2)).squeeze(2)
        attention = (query @ keys_projected.T) / math.sqrt(self.memory_size)
        attention = torch.softmax(attention, dim=1)

        sigmoid_attention = torch.sigmoid(attention)
        gate = self.gating_alpha * sigmoid_attention + self.gating_beta
        gate = F.relu(torch.tanh(gate))
        gate = gate / (gate + 1.0e-10)

        weighted_values = (
            gate.unsqueeze(2)
            * attention.unsqueeze(2)
            * values.unsqueeze(0)
        )
        if self.memory_value_aggregation == "weighted_sum":
            retrieved = weighted_values.sum(dim=1)
        else:
            # Preserve the original implementation for a tokenizer-only study.
            retrieved = weighted_values.max(dim=1)[0]
        return retrieved, keys, values

    def forward(self, input_dict):
        xyz = input_dict["partial_points"]
        batch_size = xyz.shape[0]
        pcn_latent, partial_memory_query = self._encode_partial(xyz)
        input_dict["partial_memory_query"] = partial_memory_query

        if self.training:
            gt_points = input_dict["gt_points"]
            gt_features = self.first_conv(gt_points.transpose(2, 1))
            complete_memory_feature = self.query_readout(gt_features)
            input_dict["complete_memory_feature"] = complete_memory_feature
            input_dict["memory_vector"] = self.memory_vector

        retrieved, keys, values = self._retrieve_memory(partial_memory_query)
        if self.training:
            input_dict["keys"] = keys
            input_dict["values"] = values
        input_dict["semantic_aware_feat"] = retrieved

        memory_residual = self.feature_align2(retrieved.unsqueeze(2)).squeeze(2)
        enhanced_latent = pcn_latent + memory_residual

        coarse = self.mlp(enhanced_latent).reshape(
            -1, self.number_coarse, 3
        )
        point_feat = coarse.unsqueeze(2).expand(
            -1, -1, self.grid_size**2, -1
        )
        point_feat = point_feat.reshape(
            -1, self.number_fine, 3
        ).transpose(2, 1)

        seed = self.folding_seed.unsqueeze(2).expand(
            batch_size, -1, self.number_coarse, -1
        )
        seed = seed.reshape(batch_size, -1, self.number_fine)

        expanded_latent = enhanced_latent.unsqueeze(2).expand(
            -1, -1, self.number_fine
        )
        fine_input = torch.cat([expanded_latent, seed, point_feat], dim=1)
        fine = self.final_conv(fine_input) + point_feat

        input_dict["coarse_points"] = coarse.contiguous()
        input_dict["rebuild_points"] = fine.transpose(1, 2).contiguous()
        input_dict["sampled_coarse"] = input_dict["rebuild_points"]
        return input_dict
