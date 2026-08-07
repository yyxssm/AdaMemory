#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# from models.model_utils import *
from models.FSCSVD_utils import cross_attention, furthest_point_sample, fps_subsample
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from models.FSCSVD_SVDFormer import SDG, local_encoder
from models.build import MODELS
from extensions.chamfer_dist import ChamferFunctionWithIdx


def chamfer(p1, p2):
    d1, d2, _, _ = ChamferFunctionWithIdx.apply(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = ChamferFunctionWithIdx.apply(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = ChamferFunctionWithIdx.apply(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = ChamferFunctionWithIdx.apply(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1

class Extensive_Branch_Layer(nn.Module):
    def __init__(self, latent_dim=1024):
        super(Extensive_Branch_Layer, self).__init__()
        self.latent_dim = latent_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Conv1d(1024, self.latent_dim, 1),
            nn.BatchNorm1d(self.latent_dim),
            nn.GELU(),
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape

        feature = self.first_conv(xyz.transpose(2, 1))  # (B,  256, N)
        # low_feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1)
        low_feature_global = F.adaptive_max_pool1d(feature, 1)
        feature = torch.cat([low_feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)
        feature_global = F.adaptive_max_pool1d(feature, 1)
        feature_global = torch.squeeze(feature_global, dim=2)
        multi_level_extensive_feature = torch.cat([torch.squeeze(low_feature_global, dim=2), feature_global], dim=1)
        return multi_level_extensive_feature


class offset_attention(nn.Module):
    def __init__(self, channels):
        super(offset_attention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class Multi_head_ExternalAttention(nn.Module):
    """
    # Input : x , an array with shape [B, N, C_in]
    # (batchsize, pixels, channels)
    # Parameter: M_K, a linearlayer
    # Parameter: M_V, a linearlayer
    # Parameter: heads number of heads
    # Output : out , an array with shape [B, N, C_in]
    """
    def __init__(self, channels, external_memory, point_num, heads):
        super(Multi_head_ExternalAttention, self).__init__()
        self.channels = channels
        self.external_memory = external_memory
        self.point_num = point_num
        self.heads = heads
        self.Q_linear = nn.Linear(self.channels + 3, self.channels)
        self.K_linear = nn.Linear(self.channels // self.heads, self.external_memory)
        self.softmax = nn.Softmax(dim=2)
        self.LNorm = nn.LayerNorm([self.heads, self.point_num, self.external_memory])
        self.V_Linear = nn.Linear(self.external_memory, self.channels // self.heads)
        self.O_Linear = nn.Linear(self.channels, self.channels)

    def forward(self, x, xyz):
        B, N, C = x.transpose(2, 1).shape
        x = torch.cat([x.transpose(2, 1), xyz], dim=2)
        x = self.Q_linear(x).contiguous().view(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attention = self.LNorm(self.softmax(self.K_linear(x)))
        output = self.O_Linear(self.V_Linear(attention).permute(0, 2, 1, 3).contiguous().view(B, N, C))
        return output.transpose(2, 1)


class Salience_Branch_Layer(nn.Module):
    def __init__(self, latent_dim=1024, point_scales=2048):
        super(Salience_Branch_Layer, self).__init__()
        self.latent_dim = latent_dim
        self.point_scales = point_scales

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        # self.EA = Multi_head_ExternalAttention(64, 64, self.point_scales, 4)
        self.OA = offset_attention(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.CA1 = offset_attention(128)
        self.CA2 = offset_attention(128)
        # self.CEA1 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        # self.CEA2 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.conv4 = nn.Conv1d(256, 256, 1)
        self.conv5 = nn.Conv1d(512, 512, 1)
        self.CA3 = offset_attention(512)
        self.CA4 = offset_attention(512)
        # self.CEA3 = Multi_head_ExternalAttention(512, 512, self.point_scales, 4)
        # self.CEA4 = Multi_head_ExternalAttention(512, 512, self.point_scales, 4)
        self.conv6 = nn.Conv1d(1024, 1024, 1)
        self.conv7 = nn.Conv1d(1024, self.latent_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(self.latent_dim)

    def forward(self, x):
        B, N, _ = x.shape
        self.xyz = x
        # x = F.leaky_relu(self.bn1(self.conv1(x.transpose(2, 1))), negative_slope=0.2)
        x = F.gelu(self.bn1(self.conv1(x.transpose(2, 1))))
        # x = self.EA(x, self.xyz)
        x = self.OA(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        # temp00_x_128 = self.CEA1(x, self.xyz)
        # temp01_x_128 = self.CEA2(temp00_x_128, self.xyz)
        temp00_x_128 = self.CA1(x)
        temp01_x_128 = self.CA2(temp00_x_128)
        x = torch.cat((temp00_x_128, temp01_x_128), dim=1)
        x_256 = F.gelu(self.bn4(self.conv4(x)))
        # low_feature_global = torch.max(x_256, dim=2, keepdim=True)[0]  # (B,  256, 1)
        low_feature_global = F.adaptive_max_pool1d(x_256, 1)
        x = torch.cat([low_feature_global.expand(-1, -1, N), x_256], dim=1)  # (B,  512, N)
        x_512 = F.gelu(self.bn5(self.conv5(x)))
        # temp00_x_256 = self.CEA3(x_512, self.xyz)
        # temp01_x_256 = self.CEA4(temp00_x_256, self.xyz)
        temp00_x_256 = self.CA3(x_512)
        temp01_x_256 = self.CA4(temp00_x_256)
        x = torch.cat((temp00_x_256, temp01_x_256), dim=1)
        x = F.gelu(self.bn6(self.conv6(x)))
        x = F.gelu(self.bn7(self.conv7(x)))
        # feature_global = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)
        feature_global = F.adaptive_max_pool1d(x, 1)
        feature_global = torch.squeeze(feature_global, dim=2)
        multi_level_salience_feature = torch.cat([torch.squeeze(low_feature_global, dim=2), feature_global], dim=1)
        return multi_level_salience_feature


class AutoEncoder(nn.Module):
    def __init__(self, channel=128, num_coarse=1024):
        super(AutoEncoder, self).__init__()
        self.channel = channel
        self.num_coarse = num_coarse
        self.Extensive_Encoder = Extensive_Branch_Layer()
        self.Salience_Encoder1 = Salience_Branch_Layer()
        self.cross_attn = cross_attention(d_model=1024+256, d_model_out=1024)
        # self.fc = nn.Linear(self.multi_global_size, self.multi_global_size, bias=True)
        # self.bn = nn.BatchNorm1d(self.multi_global_size)
        # generate coarse point cloud
        self.ps = nn.ConvTranspose1d(1024, self.channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(1024 + self.channel, self.channel*8, kernel_size=1)
        self.oa = offset_attention(1024)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(1024 + self.channel*4, 64, kernel_size=1)

    def encode_features(self, x):
        """Expose FSC's native seed sequence without changing its backbone."""
        extensive_feature = self.Extensive_Encoder(x)
        salience_feature = self.Salience_Encoder1(x)
        adaptive_feature = self.cross_attn(torch.unsqueeze(extensive_feature, dim=2),
                                           torch.unsqueeze(salience_feature, dim=2))
        x = F.gelu(self.ps(adaptive_feature))
        x = F.gelu(self.ps_refuse(torch.cat([x, adaptive_feature.repeat(1, 1, x.size(2))], 1)))
        seed_feature = self.oa(x)
        return salience_feature, adaptive_feature, seed_feature

    def decode_coarse(self, seed_feature, adaptive_feature, point_count):
        """Run the original FSC coarse decoder from native seed features."""
        if point_count != 2048:
            raise ValueError(
                "FSC's fixed 128-seed coarse decoder requires exactly "
                f"2048 input points, got {point_count}"
            )
        batch_size = seed_feature.shape[0]
        x2_d = seed_feature.reshape(
            batch_size,
            self.channel * 4,
            point_count // 8,
        )
        adaptive_feature_repeat = adaptive_feature.repeat(
            1,
            1,
            x2_d.size(2),
        )
        coarse_input = torch.cat(
            [x2_d.contiguous(), adaptive_feature_repeat.contiguous()],
            dim=1,
        )
        coarse = self.conv_out(F.gelu(self.conv_out1(coarse_input)))
        return coarse

    def forward(self, x):
        _, point_count, _ = x.shape
        salience_feature, adaptive_feature, seed_feature = self.encode_features(x)
        coarse = self.decode_coarse(
            seed_feature,
            adaptive_feature,
            point_count,
        )
        # output = self.FC(output)
        return salience_feature, adaptive_feature, coarse

@MODELS.register_module()
class FSCSVD(nn.Module):
    def __init__(self, cfg):
    # def __init__(self):
        super(FSCSVD, self).__init__()
        self.Encoder = AutoEncoder()
        self.merge_points = cfg.merge_points
        self.localencoder = local_encoder(cfg)
        self.refine1 = SDG(ratio=cfg.step1,hidden_dim=768, dataset=cfg.TEST_DATASET)
        self.refine2 = SDG(ratio=cfg.step2,hidden_dim=512, dataset=cfg.TEST_DATASET)
        # self.merge_points = 512
        # self.localencoder = local_encoder()
        # self.refine1 = SDG(ratio=4, hidden_dim=768, dataset='ShapeNet')
        # self.refine2 = SDG(ratio=8, hidden_dim=512, dataset='ShapeNet')

    def decode_completion(self, partial, adaptive_feature, coarse):
        """Run FSC's unchanged two-stage SDG completion decoder."""
        coarse_merge = torch.cat([partial.transpose(1, 2), coarse], dim=2)
        coarse_merge = gather_points(coarse_merge, furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points))
        local_feat = self.localencoder(coarse_merge)
        fine1 = self.refine1(local_feat, coarse_merge, adaptive_feature, partial)
        fine2 = self.refine2(local_feat, fine1, adaptive_feature, partial)
        # coarse, fine = self.Decoder(x)
        # return coarse, fine, fine
        return coarse.transpose(1, 2).contiguous(), fine1.transpose(1, 2).contiguous(), fine2.transpose(1, 2).contiguous()

    def forward(self, partial):
        _, adaptive_feature, coarse = self.Encoder(partial)
        return self.decode_completion(partial, adaptive_feature, coarse)


    def get_loss(self, pcds_pred, gt, sqrt=True,alpha1=1,alpha2=1):
        """loss function
        Args
            pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
        """
        if sqrt:
            CD = chamfer_sqrt
            PM = chamfer_single_side_sqrt
        else:
            CD = chamfer
            PM = chamfer_single_side

        Pc, P1, P2 = pcds_pred

        gt_1 = fps_subsample(gt, P1.shape[1])
        gt_c = fps_subsample(gt_1, Pc.shape[1])

        cdc = CD(Pc, gt_c)
        cd1 = CD(P1, gt_1)
        cd2 = CD(P2, gt)
        # partial_matching = PM(partial, P2)

        loss_all = cdc + alpha1*cd1 + alpha2*cd2
        losses = [cdc, cd1, cd2]
        return loss_all, losses

if __name__ == "__main__":
    input = torch.rand(size=[16, 2048, 3])
    input = input.cuda()
    model = Model()
    model = model.cuda()
    coarse, fine1, fine2 = model(input)
    print(coarse.shape, fine1.shape, fine2.shape)
    print(coarse.shape, fine1.shape, fine2.shape)
