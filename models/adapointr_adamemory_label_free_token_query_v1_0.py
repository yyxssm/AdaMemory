"""Native AdaMemory AdaPoinTr with label-free memory clustering.

PCN is referenced only for nearest-prototype pseudo labels. Class-token
insertion, memory retrieval, decoder-token fusion, and the reconstruction head
follow the repository's AdaMemory AdaPoinTr main model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from extensions.chamfer_dist import ChamferDistanceL1
from utils import misc

from .AdaMemoryAdaPoinTr import Fold, PCTransformer, SimpleRebuildFCLayer
from .build import MODELS
from .circle_loss import CircleLoss
from .label_free_adamemory_v1_0 import LabelFreeAdaMemoryMixin
from .Transformer_utils import index_points, knn_point


class LabelFreeAdaMemoryPCTransformer(PCTransformer):
    """Native AdaMemory transformer with category reads removed."""

    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_dict):
        xyz = input_dict["partial_points"]
        batch_size = xyz.shape[0]

        coordinates, features = self.grouper(xyz, self.center_num)
        features = features * (
            1.0
            / math.sqrt(256 * 128)
            / features.norm(float("inf"), dim=[1, 2], keepdim=True).clamp_min(1.0e-8)
        )
        position = self.pos_embed(coordinates)
        patches = self.input_proj(features)

        class_token = repeat(
            self.cls_token,
            "1 1 d -> b 1 d",
            b=batch_size,
        )
        encoded = torch.cat((class_token, patches), dim=1)
        encoded[:, :1] = encoded[:, :1] + self.encoder_pos_embedding
        encoded[:, 1:] = encoded[:, 1:] + position
        encoded = self.encoder(encoded, coordinates)

        global_feature = self.increase_dim(encoded).max(dim=1).values
        encoder_class_token = encoded[:, :1]
        input_dict["encoder_cls_token"] = encoder_class_token

        if self.training:
            _, complete_features = self.grouper(
                input_dict["gt_points"], self.center_num
            )
            complete_features = complete_features * (
                1.0
                / math.sqrt(256 * 128)
                / complete_features.norm(
                    float("inf"), dim=[1, 2], keepdim=True
                ).clamp_min(1.0e-8)
            )
            complete_patches = self.input_proj(complete_features)
            input_dict["gt_points_token"] = complete_patches.max(dim=1).values
            input_dict["keys"] = self.memory_vector[:, : self.encoder_embed_dim]
            input_dict["values"] = self.memory_vector[:, -self.decoder_embed_dim :]
            input_dict["memory_vector"] = self.memory_vector

        keys = self.memory_vector[:, : self.encoder_embed_dim]
        values = self.memory_vector[:, -self.decoder_embed_dim :]
        memory_query = self.memory_query_mlp(
            encoder_class_token.transpose(1, 2)
        ).squeeze(2)
        memory_keys = self.memory_key_mlp(keys.unsqueeze(2)).squeeze(2)
        memory_values = self.memory_value_mlp(values.unsqueeze(2)).squeeze(2)
        attention = torch.softmax(
            (memory_query @ memory_keys.T) / math.sqrt(self.memory_size),
            dim=1,
        )
        gate = self.gating_alpha * attention + self.gating_beta
        gate = F.relu(torch.tanh(gate))
        gate = gate / (gate + 1.0e-10)
        semantic_feature = (
            gate.unsqueeze(2)
            * attention.unsqueeze(2)
            * memory_values.unsqueeze(0)
        ).sum(dim=1)
        input_dict["semantic_aware_feat"] = semantic_feature

        fused_class_token = encoded[:, 0] + semantic_feature
        global_feature = self.global_feature_align(
            torch.cat((global_feature, fused_class_token), dim=1)
        )
        coarse = self.coarse_pred(global_feature).reshape(batch_size, -1, 3)
        coarse_input = misc.fps(xyz, self.num_query // 2)
        input_dict["input_fps"] = coarse_input
        coarse = misc.fps(torch.cat((coarse, coarse_input), dim=1), self.num_query)
        input_dict["sampled_coarse"] = coarse

        memory = self.mem_link(encoded)
        if self.training:
            picked_points = misc.jitter_points(misc.fps(xyz, 64))
            coarse = torch.cat((coarse, picked_points), dim=1)
            denoise_length = 64
        else:
            denoise_length = 0

        query = self.mlp_query(
            torch.cat(
                (
                    global_feature.unsqueeze(1).expand(-1, coarse.shape[1], -1),
                    coarse,
                ),
                dim=-1,
            )
        )
        query = torch.cat((fused_class_token.unsqueeze(1), query), dim=1)
        query = self.decoder(
            q=query,
            v=memory,
            q_pos=coarse,
            v_pos=coordinates,
            denoise_length=denoise_length or None,
        )
        input_dict["decoder_cls_token"] = query[:, :1]
        input_dict["q"] = query[:, 1:]
        input_dict["coarse_points"] = coarse
        input_dict["denoise_length"] = denoise_length
        return input_dict


@MODELS.register_module()
class AdaPoinTrAdaMemoryLabelFreeTokenQueryV1(
    LabelFreeAdaMemoryMixin,
    nn.Module,
):
    """AdaMemory AdaPoinTr main architecture without category supervision."""

    def __init__(self, config, **kwargs):
        del kwargs
        super().__init__()
        self.config = config
        self.trans_dim = int(config.decoder_config.embed_dim)
        self.encoder_dim = int(config.encoder_config.embed_dim)
        self.num_query = int(config.num_query)
        self.num_points = int(config.num_points)
        self.decoder_type = config.decoder_type
        if self.decoder_type not in {"fold", "fc"}:
            raise ValueError(
                f"Unexpected decoder_type {self.decoder_type!r}"
            )

        self.fold_step = 8
        self.base_model = LabelFreeAdaMemoryPCTransformer(config)
        if self.decoder_type == "fold":
            self.factor = self.fold_step**2
            self.decode_head = Fold(
                self.trans_dim,
                step=self.fold_step,
                hidden_dim=256,
            )
        else:
            if self.num_points % self.num_query != 0:
                raise ValueError("num_points must be divisible by num_query")
            self.factor = self.num_points // self.num_query
            self.decode_head = SimpleRebuildFCLayer(
                self.trans_dim * 2,
                step=self.factor,
            )

        # Match the native AdaMemory reconstruction head: the decoded global
        # feature, retrieved semantic feature, query, and coarse xyz all enter.
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
        )
        self.reduce_map = nn.Linear(
            self.trans_dim + 1027 + self.trans_dim,
            self.trans_dim,
        )
        self.global_feature_align = nn.Sequential(
            nn.Linear(self.trans_dim + 1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
        )
        self.memory_key_dim = self.encoder_dim
        self.memory_value_dim = self.trans_dim
        self.loss_func = ChamferDistanceL1()
        self.loss_circle_loss = CircleLoss(
            m=config.memory_circle_loss_m,
            gamma=config.memory_circle_loss_gamma,
        )

    @property
    def memory_vector(self):
        return self.base_model.memory_vector

    @property
    def gating_alpha(self):
        return self.base_model.gating_alpha

    @property
    def gating_beta(self):
        return self.base_model.gating_beta

    @property
    def query_readout(self):
        """Expose the native encoder that owns the prepended class token."""
        return self.base_model

    def get_rebuild_loss(self, input_dict, epoch=0):
        del epoch
        gt_points = input_dict["gt_points"]
        coarse_points = input_dict["coarse_points"]
        rebuild_points = input_dict["rebuild_points"]

        if "denoised_coarse" in input_dict:
            denoised_coarse = input_dict["denoised_coarse"]
            denoised_fine = input_dict["denoised_fine"]
            index = knn_point(self.factor, gt_points, denoised_coarse)
            denoised_target = index_points(gt_points, index).reshape(
                gt_points.shape[0], -1, 3
            )
            if denoised_target.shape[1] != denoised_fine.shape[1]:
                raise ValueError(
                    "Denoising prediction and target sizes do not match"
                )
            denoised_loss = self.loss_func(
                denoised_fine, denoised_target
            ) * 0.5
        else:
            denoised_loss = gt_points.new_zeros(())

        coarse_loss = self.loss_func(coarse_points, gt_points)
        fine_loss = self.loss_func(rebuild_points, gt_points)
        return denoised_loss, coarse_loss, fine_loss

    def forward(self, input_dict):
        input_dict = self.base_model(input_dict)
        if self.training:
            self._record_memory_training_features(
                input_dict,
                input_dict["encoder_cls_token"],
                input_dict["gt_points_token"],
            )

        query = input_dict["q"]
        coarse_points = input_dict["coarse_points"]
        denoise_length = int(input_dict["denoise_length"])
        decoder_class_token = input_dict["decoder_cls_token"]
        batch_size, query_count, _ = query.shape

        global_feature = self.increase_dim(
            torch.cat((decoder_class_token, query), dim=1).transpose(1, 2)
        ).transpose(1, 2).max(dim=1).values
        global_feature = self.global_feature_align(
            torch.cat((global_feature, decoder_class_token[:, 0]), dim=1)
        )
        semantic_feature = input_dict["semantic_aware_feat"]
        rebuild_feature = torch.cat(
            (
                global_feature.unsqueeze(1).expand(
                    -1, query_count, -1
                ),
                semantic_feature.unsqueeze(1).expand(
                    -1, query_count, -1
                ),
                query,
                coarse_points,
            ),
            dim=-1,
        )

        if self.decoder_type == "fold":
            rebuild_feature = self.reduce_map(
                rebuild_feature.reshape(batch_size * query_count, -1)
            )
            relative_xyz = self.decode_head(rebuild_feature).reshape(
                batch_size, query_count, 3, -1
            )
            rebuild_points = (
                relative_xyz + coarse_points.unsqueeze(-1)
            ).transpose(2, 3)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            relative_xyz = self.decode_head(rebuild_feature)
            rebuild_points = relative_xyz + coarse_points.unsqueeze(-2)

        if self.training:
            prediction_count = query_count - denoise_length
            input_dict["coarse_points"] = coarse_points[
                :, :prediction_count
            ].contiguous()
            input_dict["rebuild_points"] = rebuild_points[
                :, :prediction_count
            ].reshape(batch_size, -1, 3).contiguous()
            input_dict["denoised_coarse"] = coarse_points[
                :, prediction_count:
            ].contiguous()
            input_dict["denoised_fine"] = rebuild_points[
                :, prediction_count:
            ].reshape(batch_size, -1, 3).contiguous()
            if input_dict["rebuild_points"].shape[1] != self.num_points:
                raise ValueError("AdaPoinTr reconstruction size is incorrect")
        else:
            input_dict["coarse_points"] = coarse_points.contiguous()
            input_dict["rebuild_points"] = rebuild_points.reshape(
                batch_size, -1, 3
            ).contiguous()

        return input_dict
