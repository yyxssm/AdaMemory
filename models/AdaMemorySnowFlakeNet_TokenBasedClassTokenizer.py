##############################################################
# % Author: Yuxiang Yan
# % Date:2024/08/10
###############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_PM
from .SnowFlakeNet_utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer, grouping_operation, query_knn
from .build import MODELS
from .diffpool_gnn_pyg import GNN
from typing import Tuple
from torch import Tensor
from models.circle_loss import CircleLoss
from einops import repeat


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=True):
        super(MLP_CONV, self).__init__()
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


class TransformerClsToken(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(TransformerClsToken, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)
        
        self.encoder_pos_embedding = nn.Parameter(torch.randn(1, dim, 1))
        self.cls_token = nn.Parameter(torch.randn(1, dim, 1))

    def forward(self, x, pos):
        """
        Feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """
        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        
        cls_tokens = repeat(self.cls_token, '1 d 1 -> b d 1', b=b)
        # Mimicking ViT's operation
        x = torch.cat((cls_tokens, x), dim=2)
        # Adding a positional embedding to the class token
        
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key_x = grouping_operation(key[..., 1:].contiguous(), idx_knn)  # b, dim, n, n_knn
        qk_rel = query[..., 1:].reshape((b, -1, n, 1)) - key_x

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        qk_cls_token = (query[..., 0:1] + self.encoder_pos_embedding).unsqueeze(3).repeat(1, 1, 1, self.n_knn)
        qk_rel_x = qk_rel + pos_embedding
        qk_all = torch.cat((qk_cls_token, qk_rel_x), dim=2)
        
        attention = self.attn_mlp(qk_all)
        attention = torch.softmax(attention, -1)

        value_cls_token = (value[..., 0:1] + self.encoder_pos_embedding).unsqueeze(3).repeat(1, 1, 1, self.n_knn)
        value_x = value[..., 1:].reshape((b, -1, n, 1)) + pos_embedding
        value_all = torch.cat((value_cls_token, value_x), dim=2)
        
        agg = einsum('b c i j, b c i j -> b c i', attention, value_all)  # b, dim, n
        y = self.linear_end(agg)

        return y[..., 1:] + identity, y[..., 0:1]

def cos_similar(p, q):
    """
    Compute cosine similarity between tensors p and q.
    Args:
        p, q: Tensors with the last dimension as the feature dimension.
    Returns:
        sim_matrix: Tensor where sim_matrix[i][j] is the cosine similarity between p[i] and q[j].
    """
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix

def convert_label_to_similarity(similarity_matrix: Tensor, label_matrix: Tensor) -> Tuple[Tensor, Tensor]:
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return sub_pc

class SkipTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, key, query, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud"""
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n
        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class FeatureExtractorClsToken(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud"""
        super(FeatureExtractorClsToken, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = TransformerClsToken(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = TransformerClsToken(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

        self.cls_token_mlp = MLP_CONV(in_channel=128 + 256, layer_dims=[out_dim])

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n
        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points, l1_cls_token = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points, l2_cls_token = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        cls_token = torch.cat((l1_cls_token, l2_cls_token), dim=1)
        cls_token = self.cls_token_mlp(cls_token)

        return l3_points, cls_token


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        out_dim = 512
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, out_dim])

        self.skip_transformer = SkipTransformer(in_channel=out_dim, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=out_dim, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)  # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=out_dim * 2 + 128, hidden_dim=128, out_dim=out_dim)

        self.mlp_delta = MLP_CONV(in_channel=out_dim, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, K_prev=None, semantic_aware_feat=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)
        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        semantic_aware_feat = semantic_aware_feat.repeat(1, 1, H_up.shape[2])
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up, semantic_aware_feat], 1))
        # K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, K_curr


class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(Decoder, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat, partial, return_P0=False, semantic_aware_feat=None):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3)     Initial coarse point cloud
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = semantic_aware_feat.repeat(1, 1, self.num_p0)
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev, semantic_aware_feat=semantic_aware_feat)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd  # arr_pcd list contains the progressively upsampled point clouds

@MODELS.register_module()
class AdaMemorySnowFlakeNet_TokenBasedClassTokenizer(nn.Module):
    def __init__(self, config, **kwargs):
        """
        Args:
            dim_feat: int, dimension of global feature
            num_pc: int
            num_p0: int
            radius: searching radius
            up_factors: list of int
        """
        super().__init__()
        self.config = config
        dim_feat = config.dim_feat
        num_pc = config.num_pc
        num_p0 = config.num_p0
        radius = config.radius
        up_factors = config.up_factors

        self.feat_extractor_cls_token = FeatureExtractorClsToken(out_dim=dim_feat)
        self.decoder = Decoder(dim_feat=dim_feat, num_pc=num_pc, num_p0=num_p0, radius=radius, up_factors=up_factors)
        
        self.memory_size = config.get("memory_size", config.class_num)
        self.gating_alpha = nn.Parameter(torch.randn(1, self.memory_size))
        self.gating_beta = nn.Parameter(torch.randn(1, self.memory_size))
        self.encoder_embed_dim = config.encoder_config.embed_dim
        self.decoder_embed_dim = config.decoder_config.embed_dim
        self.memory_vector = nn.Parameter(torch.rand((self.memory_size, self.encoder_embed_dim + self.decoder_embed_dim)))
        self.memory_query_mlp = MLP_CONV(in_channel=config.encoder_config.embed_dim, layer_dims=[384, config.encoder_config.embed_dim])
        self.memory_key_mlp = MLP_CONV(in_channel=config.encoder_config.embed_dim, layer_dims=[384, config.encoder_config.embed_dim])
        
        self.feature_align = MLP_CONV(in_channel=1024, layer_dims=[384, 512])
        
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func_CD = ChamferDistanceL1()
        self.loss_func_PM = ChamferDistanceL1_PM()
        self.loss_circle_loss = CircleLoss(m=self.config.memory_circle_loss_m,
                                           gamma=self.config.memory_circle_loss_gamma)

    def get_rebuild_loss(self, input_dict, epoch=1, **kwargs):
        """
        Loss function for reconstruction.
        Args:
            input_dict: Dictionary containing input data.
        Returns:
            loss_coarse, loss_fine: Coarse and fine reconstruction losses.
        """
        if len(input_dict["coarse_points"]) == 3:
            Pc, P1, P2 = input_dict["coarse_points"]
            P3 = input_dict["rebuild_points"]
            complete_gt = input_dict["gt_points"]
            partial_input = input_dict["partial_points"]

            gt_2 = fps(complete_gt, P2.shape[1])
            gt_1 = fps(gt_2, P1.shape[1])
            gt_c = fps(gt_1, Pc.shape[1])

            cdc = self.loss_func_CD(Pc, gt_c)
            cd1 = self.loss_func_CD(P1, gt_1)
            cd2 = self.loss_func_CD(P2, gt_2)
            cd3 = self.loss_func_CD(P3, complete_gt)

            partial_matching = self.loss_func_PM(partial_input, P3)
            
            loss_coarse = cdc + cd1 + cd2 + partial_matching
            loss_fine = cd3
        elif len(input_dict["coarse_points"]) == 4:
            Pc, P0, P1, P2 = input_dict["coarse_points"]
            P3 = input_dict["rebuild_points"]
            complete_gt = input_dict["gt_points"]
            partial_input = input_dict["partial_points"]

            gt_2 = fps(complete_gt, P2.shape[1])
            gt_1 = fps(gt_2, P1.shape[1])
            gt_0 = fps(gt_1, P0.shape[1])
            gt_c = fps(gt_0, Pc.shape[1])

            cdc = self.loss_func_CD(Pc, gt_c)
            cd0 = self.loss_func_CD(P0, gt_0)
            cd1 = self.loss_func_CD(P1, gt_1)
            cd2 = self.loss_func_CD(P2, gt_2)
            cd3 = self.loss_func_CD(P3, complete_gt)

            partial_matching = self.loss_func_PM(partial_input, P3)
            
            loss_coarse = cdc + cd0 + cd1 + cd2 + partial_matching
            loss_fine = cd3

        return torch.tensor(0).cuda(), loss_coarse, loss_fine

    def get_compactness_loss(self, input_dict):
        """
        Compactness loss calculation.
        Args:
            input_dict: Dictionary containing input data.
        Returns:
            feature_feature_circle_loss, memory_feature_loss: Loss values.
        """
        class_label = input_dict['taxonomy_ids']
        memory_vector = input_dict["memory_vector"]
        encoder_cls_token = input_dict["encoder_cls_token"].transpose(1, 2)
        gt_points_token = input_dict["gt_points_token"].transpose(1, 2)
        lowest_num = self.config.get("not_similar_loss_num", 1)
        positive_thres = self.config.memory_circle_loss_m
        negative_thres = 1 - positive_thres
        
        cls_token = torch.concat((encoder_cls_token, gt_points_token), dim=-1).squeeze(1)
        # Compute cosine similarity
        feature_memory_cos_sim = cos_similar(cls_token, memory_vector)
        feature_feature_cos_sim = cos_similar(cls_token, cls_token)
        
        # Close the similarity between the current feature and the nearest prototype
        feature_memory_similar_score, feature_memory_idx = feature_memory_cos_sim.max(-1)
        
        memory_size, _ = memory_vector.shape
        memory_label_list = [[] for _ in range(memory_size)]
        memory_label_length_list = [[] for _ in range(memory_size)]

        for i in range(len(feature_memory_idx)):
            memory_label_list[int(feature_memory_idx[i])].append(int(class_label[i]))

        for i in range(len(memory_label_length_list)):
            memory_label_length_list[i] = len(set(memory_label_list[i]))
        
        label_flag_matrix = class_label.unsqueeze(1) == class_label.unsqueeze(0)
        memory_flag_matrix = feature_memory_idx.unsqueeze(1) == feature_memory_idx.unsqueeze(0)
        positive_flag = torch.logical_and(label_flag_matrix, memory_flag_matrix)
        feature_feature_sp, feature_feature_sn = convert_label_to_similarity(feature_feature_cos_sim, positive_flag)
        feature_feature_circle_loss = self.loss_circle_loss(feature_feature_sp, feature_feature_sn)
        
        class_num = torch.gather(torch.tensor(memory_label_length_list).cuda(), 0, feature_memory_idx)
        memory_feature_positive_loss = F.relu(positive_thres - feature_memory_similar_score) / class_num
        memory_feature_positive_loss = memory_feature_positive_loss.mean()
        
        similar_score, _ = feature_memory_similar_score.sort()
        similar_score = similar_score[..., :lowest_num]
        memory_feature_negative_loss = F.relu(similar_score - negative_thres).mean()
        
        memory_feature_loss = memory_feature_positive_loss + memory_feature_negative_loss
        
        return feature_feature_circle_loss, memory_feature_loss

    def get_sim_seperation_loss(self, input_dict):
        """
        This loss ensures that the memories are distinct enough to avoid clustering into a single category.
        Args:
            input_dict: Dictionary containing input data.
        Returns:
            key_value_cos_sim: Separation loss value.
        """
        positive_thres = self.config.memory_circle_loss_m
        negative_thres = 1 - positive_thres
        memory_vector = input_dict["memory_vector"]
        memory_size = memory_vector.shape[0]

        key_value_cos_sim = cos_similar(memory_vector, memory_vector)
        key_value_cos_sim -= torch.eye(memory_size).cuda()
        key_value_cos_sim = F.relu(key_value_cos_sim - negative_thres)
        key_value_cos_sim = key_value_cos_sim.mean() / 2  # Since the similarity matrix is symmetric, only calculate once

        return key_value_cos_sim

    def forward(self, input_dict, return_P0=False):
        """
        Forward pass for the network.
        Args:
            input_dict: Dictionary containing input data.
            return_P0: Boolean flag to indicate if P0 should be returned.
        Returns:
            input_dict: Dictionary containing output data.
        """
        point_cloud = input_dict["partial_points"]
        adj = input_dict["adj"]
        
        if self.training:
            _, gt_cls_token = self.feat_extractor_cls_token(input_dict["gt_points"].transpose(2, 1).contiguous())  # B 512 n
            gt_cls_token = gt_cls_token * (1 / torch.sqrt(torch.tensor(512)) / gt_cls_token.norm(float('inf'), dim=[1], keepdim=True))
            input_dict["gt_points_token"] = gt_cls_token
            
            keys_reparam = self.memory_vector[:, :self.encoder_embed_dim]
            values_reparam = self.memory_vector[:, -self.decoder_embed_dim:]
        else:
            keys_reparam = self.memory_vector[:, :self.encoder_embed_dim]
            values_reparam = self.memory_vector[:, -self.decoder_embed_dim:]
        
        input_dict["keys"] = keys_reparam
        input_dict["values"] = values_reparam
        input_dict["memory_vector"] = self.memory_vector
        
        pcd_bnc = point_cloud
        point_cloud = point_cloud.permute(0, 2, 1).contiguous()
        feat, cls_token = self.feat_extractor_cls_token(point_cloud)  # b c num_points
        cls_token = cls_token * (1 / torch.sqrt(torch.tensor(512)) / cls_token.norm(float('inf'), dim=[1], keepdim=True))
        input_dict["encoder_cls_token"] = cls_token
        
        memory_q = self.memory_query_mlp(feat).squeeze(2)
        memory_k = self.memory_key_mlp(keys_reparam.unsqueeze(2)).squeeze(2)

        attn = (memory_q @ memory_k.T) / torch.sqrt(torch.tensor(self.memory_size))
        attn = torch.softmax(attn, dim=1)
        gating_param = self.gating_alpha * attn + self.gating_beta
        gating_param = F.relu(torch.tanh(gating_param))
        gating_param = gating_param / (gating_param + 1e-10)  # Convert to indicator function
        
        semantic_aware_feat = (gating_param.unsqueeze(2) * attn.unsqueeze(2) * values_reparam.unsqueeze(0)).max(1)[0].unsqueeze(2)  # b c 1
        input_dict["semantic_aware_feat"] = semantic_aware_feat
        
        feat = torch.cat((feat, cls_token), dim=1)
        feat = self.feature_align(feat)
        
        out = self.decoder(feat, pcd_bnc, return_P0=return_P0, semantic_aware_feat=semantic_aware_feat)
        if self.training:
            input_dict["coarse_points"] = out[:-1]
            input_dict["rebuild_points"] = out[-1]
            input_dict["sampled_coarse"] = input_dict["rebuild_points"]  # Placeholder, no function
        else:
            input_dict["coarse_points"] = out[1]
            input_dict["rebuild_points"] = out[-1]  # out[-1] is the final generated point cloud
            input_dict["sampled_coarse"] = input_dict["rebuild_points"]  # Placeholder, no function
        return input_dict
