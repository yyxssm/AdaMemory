"""Shared label-free clustering losses for native AdaMemory models.

This module deliberately has no category-label or classification dependency.
The clustering target is induced exclusively by the nearest learnable memory
prototype assigned to each paired incomplete/complete feature.
"""

import torch
import torch.nn.functional as F


def cosine_similarity(p, q, eps=1.0e-8):
    """Return pairwise cosine similarity with finite zero-vector handling."""
    p = F.normalize(p, p=2, dim=-1, eps=eps)
    q = F.normalize(q, p=2, dim=-1, eps=eps)
    return p.matmul(q.transpose(-2, -1))


class LabelFreeAdaMemoryMixin:
    """Nearest-prototype clustering and prototype-separation objectives."""

    def _record_memory_training_features(
        self,
        input_dict,
        partial_memory_query,
        complete_memory_feature,
    ):
        input_dict["partial_memory_query"] = partial_memory_query
        input_dict["complete_memory_feature"] = complete_memory_feature

    @staticmethod
    def _flatten_memory_feature(feature, expected_dim, name):
        """Accept both native [B, 1, D] and [B, D, 1] token layouts."""
        if feature.ndim == 3:
            if feature.shape[1] == 1:
                feature = feature.squeeze(1)
            elif feature.shape[2] == 1:
                feature = feature.squeeze(2)
        if feature.ndim != 2 or feature.shape[1] != expected_dim:
            raise ValueError(
                f"{name} must flatten to [B, {expected_dim}], got "
                f"{tuple(feature.shape)}"
            )
        return feature

    def _nearest_assignment_circle_loss(
        self,
        paired_feature,
        nearest_index,
    ):
        """Match the PCN main-experiment label-free clustering objective.

        Two samples are positives when their incomplete/complete feature pairs
        choose the same nearest memory prototype.  Samples assigned to
        different prototypes are negatives.  Category labels never enter this
        construction.
        """
        feature_similarity = cosine_similarity(paired_feature, paired_feature)
        positive_pair_mask = nearest_index.unsqueeze(1).eq(
            nearest_index.unsqueeze(0)
        )
        positive_mask = positive_pair_mask.triu(diagonal=1)
        negative_mask = positive_pair_mask.logical_not().triu(diagonal=1)
        positive_similarity = feature_similarity[positive_mask]
        negative_similarity = feature_similarity[negative_mask]
        return self.loss_circle_loss(
            positive_similarity,
            negative_similarity,
        )

    def get_compactness_loss(self, input_dict):
        """Cluster paired features using nearest-memory assignments only."""
        partial_query = self._flatten_memory_feature(
            input_dict["partial_memory_query"],
            self.memory_key_dim,
            "partial_memory_query",
        )
        complete_feature = self._flatten_memory_feature(
            input_dict["complete_memory_feature"],
            self.memory_value_dim,
            "complete_memory_feature",
        )

        paired_feature = torch.cat(
            (partial_query, complete_feature), dim=-1
        )
        if paired_feature.shape[-1] != self.memory_vector.shape[-1]:
            raise ValueError(
                "Paired feature and memory dimensions differ: "
                f"{paired_feature.shape[-1]} vs {self.memory_vector.shape[-1]}"
            )

        feature_memory_similarity = cosine_similarity(
            paired_feature, self.memory_vector
        )
        nearest_score, nearest_index = feature_memory_similarity.max(dim=-1)
        feature_circle_loss = self._nearest_assignment_circle_loss(
            paired_feature,
            nearest_index,
        )

        positive_threshold = float(self.config.memory_circle_loss_m)
        negative_threshold = 1.0 - positive_threshold
        memory_positive_loss = F.relu(
            positive_threshold - nearest_score
        ).mean()

        lowest_num = max(
            1,
            min(
                int(self.config.get("not_similar_loss_num", 1)),
                nearest_score.numel(),
            ),
        )
        low_similarity = nearest_score.sort().values[:lowest_num]
        memory_negative_loss = F.relu(
            low_similarity - negative_threshold
        ).mean()
        memory_feature_loss = memory_positive_loss + memory_negative_loss
        return feature_circle_loss, memory_feature_loss

    def get_sim_seperation_loss(self, input_dict):
        """Keep memory prototypes separated; spelling follows runner API."""
        del input_dict
        memory_vector = self.memory_vector
        negative_threshold = 1.0 - float(self.config.memory_circle_loss_m)
        memory_similarity = cosine_similarity(memory_vector, memory_vector)
        identity = torch.eye(
            memory_similarity.shape[0],
            device=memory_similarity.device,
            dtype=memory_similarity.dtype,
        )
        off_diagonal_similarity = memory_similarity - identity
        return F.relu(
            off_diagonal_similarity - negative_threshold
        ).mean() / 2.0
