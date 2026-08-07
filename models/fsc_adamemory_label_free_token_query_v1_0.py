"""FSC with a 256-D, category-label-free AdaMemory query plugin.

The original FSC encoder, coarse generator, and two SDG refinement stages keep
their module names and execution order. AdaMemory only reads FSC's existing
attention-enhanced seed sequence, retrieves a complete-shape prior, and adds
that prior to FSC's global latent feature before decoding.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .build import MODELS
from .FSCSVD import FSCSVD
from .FSCSVD_utils import self_attention


class _AllGatherWithGradient(torch.autograd.Function):
    """All-gather equal local batches while preserving encoder gradients."""

    @staticmethod
    def forward(ctx, tensor):
        if not dist.is_available() or not dist.is_initialized():
            ctx.world_size = 1
            ctx.rank = 0
            return (tensor,)

        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        gathered = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.world_size == 1:
            return grad_outputs[0]

        # Every process evaluates the global gathered batch. Reduce gradients
        # for each source rank independently, then return this rank's row.
        gathered_grads = torch.stack(
            [gradient.contiguous() for gradient in grad_outputs],
            dim=0,
        )
        dist.all_reduce(gathered_grads, op=dist.ReduceOp.SUM)
        return gathered_grads[ctx.rank]


def _gather_training_features(feature):
    return torch.cat(_AllGatherWithGradient.apply(feature), dim=0)


def _cosine_similarity(lhs, rhs, eps=1.0e-8):
    lhs = F.normalize(lhs, p=2, dim=-1, eps=eps)
    rhs = F.normalize(rhs, p=2, dim=-1, eps=eps)
    return lhs @ rhs.transpose(-2, -1)


class _PointwiseProjection(nn.Module):
    """A small Q/K/V projection that preserves its input/output width."""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, feature_dim, 1),
        )

    def forward(self, feature):
        return self.net(feature)


class FSCSeedClassTokenizer(nn.Module):
    """Project FSC's 1024-D seeds and aggregate them into a 256-D query."""

    seed_input_dim = 1024

    def __init__(
        self,
        token_dim=256,
        depth=1,
        num_heads=8,
        dim_feedforward=1024,
        drop_rate=0.0,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        if depth < 1:
            raise ValueError("class_tokenizer.depth must be positive")
        if self.token_dim <= 0:
            raise ValueError("class_tokenizer.token_dim must be positive")
        if self.token_dim % num_heads != 0:
            raise ValueError(
                "class_tokenizer.token_dim must be divisible by num_heads"
            )

        self.seed_projection = nn.Conv1d(
            self.seed_input_dim,
            self.token_dim,
            kernel_size=1,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, self.token_dim, 1))
        self.blocks = nn.ModuleList(
            [
                self_attention(
                    d_model=self.token_dim,
                    d_model_out=self.token_dim,
                    nhead=int(num_heads),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(drop_rate),
                )
                for _ in range(int(depth))
            ]
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, seed_features):
        if (
            seed_features.ndim != 3
            or seed_features.shape[1] != self.seed_input_dim
        ):
            raise ValueError(
                "seed_features must have shape [B, 1024, S], got "
                f"{tuple(seed_features.shape)}"
            )

        seed_features = self.seed_projection(seed_features)
        cls_token = self.cls_token.expand(seed_features.shape[0], -1, -1)
        tokens = torch.cat((cls_token, seed_features), dim=2)
        for block in self.blocks:
            tokens = block(tokens)

        cls_token = tokens[:, :, 0]
        return self.output_norm(cls_token).unsqueeze(2).contiguous()


@MODELS.register_module()
class FSCSVDAdaMemoryLabelFreeTokenQueryV1(FSCSVD):
    """Original FSC plus the label-free AdaMemory training objective.

    FSC's native 1024-channel seed sequence is projected only on the memory
    query side path. Query, key, and value are all 256-D; the value remains
    native to FSC's shared point-wise ``first_conv`` feature.
    """

    memory_key_dim = 256
    memory_value_dim = 256
    memory_fusion_dim = 1024

    def __init__(self, config, **kwargs):
        del kwargs

        # Construct every original FSC parameter first. With the same seed,
        # common baseline tensors therefore have exactly the same initialization.
        super().__init__(config)
        self.config = config

        self.memory_size = int(config.memory_size)
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")

        tokenizer_config = config.get("class_tokenizer", {})
        token_dim = int(
            tokenizer_config.get("token_dim", self.memory_key_dim)
        )
        if token_dim != self.memory_key_dim:
            raise ValueError(
                "FSC AdaMemory v1_0 requires class_tokenizer.token_dim=256 "
                "so query and memory key widths match"
            )
        self.query_readout = FSCSeedClassTokenizer(
            token_dim=token_dim,
            depth=int(tokenizer_config.get("depth", 1)),
            num_heads=int(tokenizer_config.get("num_heads", 8)),
            dim_feedforward=int(
                tokenizer_config.get("dim_feedforward", 1024)
            ),
            drop_rate=float(tokenizer_config.get("drop_rate", 0.0)),
        )

        memory_width = self.memory_key_dim + self.memory_value_dim
        self.memory_vector = nn.Parameter(
            torch.empty(self.memory_size, memory_width)
        )
        nn.init.normal_(self.memory_vector, std=0.02)

        self.gating_alpha = nn.Parameter(torch.randn(1, self.memory_size))
        self.gating_beta = nn.Parameter(torch.randn(1, self.memory_size))
        projection_hidden_dim = int(
            config.get("memory_projection_hidden_dim", 384)
        )
        self.memory_query_mlp = _PointwiseProjection(
            self.memory_key_dim,
            projection_hidden_dim,
        )
        self.memory_key_mlp = _PointwiseProjection(
            self.memory_key_dim,
            projection_hidden_dim,
        )
        self.memory_value_mlp = _PointwiseProjection(
            self.memory_value_dim,
            projection_hidden_dim,
        )

        # Bias-free alignment preserves the paper's exact residual fallback:
        # a fully closed gate maps to a zero residual.
        self.memory_value_to_latent = nn.Sequential(
            nn.Conv1d(
                self.memory_value_dim,
                512,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(
                512,
                self.memory_fusion_dim,
                kernel_size=1,
                bias=False,
            ),
        )

        self.memory_circle_margin = float(config.memory_circle_loss_m)
        self.memory_circle_gamma = float(config.memory_circle_loss_gamma)
        self.gating_eps = float(config.get("gating_eps", 1.0e-10))
        self.gating_temperature = float(
            config.get("gating_temperature", 1.0)
        )
        if self.gating_eps <= 0.0:
            raise ValueError("gating_eps must be positive")
        if self.gating_temperature <= 0.0:
            raise ValueError("gating_temperature must be positive")
        self.scatter_reduction = str(
            config.get("scatter_reduction", "mean")
        )
        if self.scatter_reduction not in {"mean", "sum"}:
            raise ValueError("scatter_reduction must be 'mean' or 'sum'")

    @property
    def memory_keys(self):
        return self.memory_vector[:, : self.memory_key_dim]

    @property
    def memory_values(self):
        return self.memory_vector[:, self.memory_key_dim :]

    def _retrieve_memory(self, memory_query):
        query = self.memory_query_mlp(memory_query).squeeze(2)
        keys = self.memory_key_mlp(self.memory_keys.unsqueeze(2)).squeeze(2)
        values = self.memory_value_mlp(
            self.memory_values.unsqueeze(2)
        ).squeeze(2)

        # The paper defines w = softmax(QK^T), without memory-size scaling.
        attention = torch.softmax(query @ keys.transpose(0, 1), dim=1)
        gate_logits = (
            self.gating_alpha * torch.sigmoid(attention)
            + self.gating_beta
        )
        gate_score = F.relu(torch.tanh(gate_logits))

        # Keep the paper's near-binary indicator in the forward pass.  Its
        # direct derivative is effectively zero in FP32 when eps is small, and
        # ReLU would also make every closed slot permanently inactive.  A
        # sigmoid straight-through surrogate supplies stable gradients to both
        # open and closed slots without changing the retrieved feature value.
        forward_indicator = gate_score / (
            gate_score + self.gating_eps
        )
        surrogate_indicator = torch.sigmoid(
            gate_logits / self.gating_temperature
        )
        gate_indicator = (
            forward_indicator.detach()
            + (
                surrogate_indicator
                - surrogate_indicator.detach()
            )
        )
        retrieved = (
            gate_indicator.unsqueeze(2)
            * attention.unsqueeze(2)
            * values.unsqueeze(0)
        ).sum(dim=1)
        return retrieved, attention, gate_score, gate_indicator

    def forward(self, input_dict):
        partial_points = input_dict["partial_points"]
        point_count = partial_points.shape[1]

        _, adaptive_feature, seed_feature = self.Encoder.encode_features(
            partial_points
        )
        memory_query = self.query_readout(seed_feature)
        retrieved, attention, gate_score, gate_indicator = (
            self._retrieve_memory(memory_query)
        )
        memory_residual = self.memory_value_to_latent(
            retrieved.unsqueeze(2)
        )
        enhanced_feature = adaptive_feature + memory_residual

        coarse = self.Encoder.decode_coarse(
            seed_feature,
            enhanced_feature,
            point_count,
        )
        coarse_points, fine1, rebuild_points = self.decode_completion(
            partial_points,
            enhanced_feature,
            coarse,
        )

        input_dict["coarse_points"] = coarse_points
        input_dict["fine1"] = fine1
        input_dict["rebuild_points"] = rebuild_points
        input_dict["partial_memory_query"] = memory_query
        input_dict["semantic_aware_feat"] = retrieved
        input_dict["memory_attention"] = attention
        input_dict["memory_gate_score"] = gate_score
        input_dict["memory_gate"] = gate_indicator

        if self.training:
            if "gt_points" not in input_dict:
                raise KeyError(
                    "gt_points is required only while training AdaMemory FSC"
                )
            complete_point_features = (
                self.Encoder.Extensive_Encoder.first_conv(
                    input_dict["gt_points"].transpose(2, 1).contiguous()
                )
            )
            input_dict["complete_memory_feature"] = F.adaptive_max_pool1d(
                complete_point_features,
                1,
            )

        return input_dict

    def get_loss(self, input_dict, sqrt=True, alpha1=1, alpha2=1):
        predictions = (
            input_dict["coarse_points"],
            input_dict["fine1"],
            input_dict["rebuild_points"],
        )
        return super().get_loss(
            predictions,
            input_dict["gt_points"],
            sqrt=sqrt,
            alpha1=alpha1,
            alpha2=alpha2,
        )

    @staticmethod
    def _flatten_feature(feature, expected_dim, name):
        if feature.ndim == 3 and feature.shape[-1] == 1:
            feature = feature.squeeze(2)
        elif feature.ndim == 3 and feature.shape[1] == 1:
            feature = feature.squeeze(1)
        if feature.ndim != 2 or feature.shape[1] != expected_dim:
            raise ValueError(
                f"{name} must flatten to [B, {expected_dim}], got "
                f"{tuple(feature.shape)}"
            )
        return feature

    def _circle_loss(self, positive_scores, negative_scores):
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            return (positive_scores.sum() + negative_scores.sum()) * 0.0

        q = self.memory_circle_margin
        gamma = self.memory_circle_gamma
        positive_weight = F.relu(positive_scores.detach() + q)
        negative_weight = F.relu(1.0 + q - negative_scores.detach())
        positive_logits = -gamma * positive_weight * positive_scores
        negative_logits = gamma * negative_weight * negative_scores
        return F.softplus(
            torch.logsumexp(positive_logits, dim=0)
            + torch.logsumexp(negative_logits, dim=0)
        )

    def _batched_circle_loss(self, similarity, positive_mask):
        """Evaluate every memory slot without a Python/CUDA sync loop."""
        if similarity.shape != positive_mask.shape:
            raise ValueError(
                "similarity and positive_mask must have the same shape, got "
                f"{tuple(similarity.shape)} and {tuple(positive_mask.shape)}"
            )

        negative_mask = ~positive_mask
        valid_slots = negative_mask.any(dim=0)
        q = self.memory_circle_margin
        gamma = self.memory_circle_gamma
        positive_weight = F.relu(similarity.detach() + q)
        negative_weight = F.relu(1.0 + q - similarity.detach())
        positive_logits = -gamma * positive_weight * similarity
        negative_logits = gamma * negative_weight * similarity

        positive_logits = positive_logits.masked_fill(
            ~positive_mask,
            -torch.inf,
        )
        negative_logits = negative_logits.masked_fill(
            ~negative_mask,
            -torch.inf,
        )
        # A slot can have no negatives (for example with a one-sample global
        # batch). Fill only those unused columns with finite constants so their
        # logsumexp backward cannot produce NaNs; valid_slots masks them out.
        negative_logits = torch.where(
            valid_slots.unsqueeze(0),
            negative_logits,
            torch.zeros_like(negative_logits),
        )
        losses = F.softplus(
            torch.logsumexp(positive_logits, dim=0)
            + torch.logsumexp(negative_logits, dim=0)
        )
        valid_weights = valid_slots.to(dtype=losses.dtype)
        return (
            losses * valid_weights
        ).sum() / valid_weights.sum().clamp_min(1.0)

    def _prototype_cluster_losses(
        self,
        partial_features,
        complete_features,
    ):
        key_similarity = _cosine_similarity(
            partial_features,
            self.memory_keys,
        )
        value_similarity = _cosine_similarity(
            complete_features,
            self.memory_values,
        )
        nearest_memory = key_similarity.argmax(dim=1)
        base_positive = key_similarity.argmax(dim=0)
        memory_indices = torch.arange(
            self.memory_size,
            device=key_similarity.device,
        )
        positive_mask = nearest_memory.unsqueeze(1).eq(
            memory_indices.unsqueeze(0)
        )
        positive_mask[base_positive, memory_indices] = True

        key_loss = self._batched_circle_loss(
            key_similarity,
            positive_mask,
        )
        # Key assignments select paired complete features for values at the
        # same slot, preserving the one-to-one key/value memory semantics.
        value_loss = self._batched_circle_loss(
            value_similarity,
            positive_mask,
        )
        return key_loss, value_loss

    def _scatter_loss(self):
        if self.memory_size < 2:
            return self.memory_vector.sum() * 0.0

        # Normalize the paired key/value halves independently before measuring
        # prototype similarity.
        paired_memory = torch.cat(
            (
                F.normalize(self.memory_keys, p=2, dim=1),
                F.normalize(self.memory_values, p=2, dim=1),
            ),
            dim=1,
        )
        paired_memory = F.normalize(paired_memory, p=2, dim=1)
        similarity = paired_memory @ paired_memory.transpose(0, 1)
        off_diagonal = ~torch.eye(
            self.memory_size,
            device=similarity.device,
            dtype=torch.bool,
        )
        penalties = F.relu(
            similarity[off_diagonal] - self.memory_circle_margin
        )
        if self.scatter_reduction == "sum":
            return penalties.sum()
        return penalties.mean()

    def get_adamemory_losses(self, input_dict):
        """Return only the paper's label-free cluster and scatter losses."""
        partial_features = self._flatten_feature(
            input_dict["partial_memory_query"],
            self.memory_key_dim,
            "partial_memory_query",
        )
        complete_features = self._flatten_feature(
            input_dict["complete_memory_feature"],
            self.memory_value_dim,
            "complete_memory_feature",
        )
        partial_features = _gather_training_features(partial_features)
        complete_features = _gather_training_features(complete_features)

        key_cluster, value_cluster = self._prototype_cluster_losses(
            partial_features,
            complete_features,
        )
        return {
            "key_cluster": key_cluster,
            "value_cluster": value_cluster,
            "cluster": 0.5 * (key_cluster + value_cluster),
            "scatter": self._scatter_loss(),
        }
