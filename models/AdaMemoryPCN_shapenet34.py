##############################################################
# % Author: Yuxiang Yan
# % Date:2024/08/10
###############################################################

import torch
import torch.nn as nn
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2
from .diffpool_gnn_pyg import GNN
from typing import Tuple
from torch import Tensor
from models.circle_loss import CircleLoss
import torch.nn.functional as F

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

def cos_similar(p, q):
    # params: Tensors p and q, ensuring the last dimension is the feature dimension, and the second-to-last dimension is the similarity dimension.
    # return: sim_matrix[i][j] represents the similarity value between the i-th feature of p and the j-th feature of q.
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

@MODELS.register_module()
class AdaMemoryPCN_shapenet34(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.number_fine = config.num_pred
        self.encoder_channel = config.encoder_channel
        grid_size = 4  # set default
        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size ** 2)
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.number_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2).cuda()  # 1 2 S

        # memory bank code
        self.memory_size = config.get("memory_size", config.class_num)
        self.gating_alpha = nn.Parameter(torch.randn(1, self.memory_size))
        self.gating_beta = nn.Parameter(torch.randn(1, self.memory_size))
        self.encoder_embed_dim = config.encoder_config.embed_dim
        self.decoder_embed_dim = config.decoder_config.embed_dim
        self.memory_vector = nn.Parameter(torch.rand((self.memory_size, self.encoder_embed_dim + self.decoder_embed_dim)))
        self.memory_query_mlp = MLP_CONV(in_channel=config.encoder_config.embed_dim, layer_dims=[384, config.encoder_config.embed_dim])
        self.memory_key_mlp = MLP_CONV(in_channel=config.encoder_config.embed_dim, layer_dims=[384, config.encoder_config.embed_dim])

        self.gnn1_embed = GNN(3, 128, 128, lin=False)
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, config.decoder_config.embed_dim)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(config.decoder_config.embed_dim)
        self.feature_align = MLP_CONV(in_channel=2048, layer_dims=[512, 1024])
        self.classifier = MLP_CONV(in_channel=config.decoder_config.embed_dim, layer_dims=[128, config.class_num])

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()
        self.loss_circle_loss = CircleLoss(m=self.config.memory_circle_loss_m,
                                           gamma=self.config.memory_circle_loss_gamma)

    # Version 3: Only fix the total number of memory banks, not the number of subclasses per category, introducing circle loss and corresponding ideas
    def get_compactness_loss(self, input_dict):
        class_label = input_dict['taxonomy_ids']
        memory_vector = input_dict["memory_vector"]
        encoder_cls_token = input_dict["encoder_cls_token"].squeeze(2)
        gt_points_token = input_dict["gt_points_token"]
        lowest_num = self.config.get("not_similar_loss_num", 1)
        positive_thres = self.config.memory_circle_loss_m
        negative_thres = 1 - positive_thres

        cls_token = torch.concat((encoder_cls_token, gt_points_token), dim=1)
        # Calculate cosine similarity
        feature_memory_cos_sim = cos_similar(cls_token, memory_vector)
        feature_feature_cos_sim = cos_similar(cls_token, cls_token)

        # Increase the similarity between the current feature and the nearest prototype
        feature_memory_similar_score, feature_memory_idx = feature_memory_cos_sim.max(-1)

        memory_size, _ = memory_vector.shape
        # Used to store which classes of features point to each memory vector
        memory_label_list = [[] for _ in range(memory_size)]
        memory_label_length_list = [[] for _ in range(memory_size)]
        # Update memory_label_list
        for i in range(len(feature_memory_idx)):
            memory_label_list[int(feature_memory_idx[i])].append(int(class_label[i]))
        # Calculate how many classes point to a certain memory vector in memory_label_list
        for i in range(len(memory_label_length_list)):
            memory_label_length_list[i] = len(set(memory_label_list[i]))

        # Introduce circle loss
        # anchor: memory vector
        # positive: same category, pointing to the same memory vector
        # negative: different category, pointing to different memory vectors

        # Circle loss for the same category and subclass
        label_flag_matrix = class_label.unsqueeze(1) == class_label.unsqueeze(0)
        memory_flag_matrix = feature_memory_idx.unsqueeze(1) == feature_memory_idx.unsqueeze(0)
        positive_flag = torch.logical_and(label_flag_matrix, memory_flag_matrix)
        # Calculate similarity between features
        feature_feature_sp, feature_feature_sn = convert_label_to_similarity(feature_feature_cos_sim, positive_flag)
        feature_feature_circle_loss = self.loss_circle_loss(feature_feature_sp, feature_feature_sn)

        # Increase the similarity between the current feature and the nearest prototype
        # Using uncertainty ideas, if multiple categories point to a memory vector, the weight should be reduced during feature similarity
        class_num = torch.gather(torch.tensor(memory_label_length_list).cuda(), 0, feature_memory_idx)
        memory_feature_positive_loss = F.relu(positive_thres - feature_memory_similar_score) / class_num
        memory_feature_positive_loss = memory_feature_positive_loss.mean()

        similar_score, _ = feature_memory_similar_score.sort()
        similar_score = similar_score[..., :lowest_num]
        memory_feature_negative_loss = F.relu(similar_score - negative_thres).mean()

        memory_feature_loss = memory_feature_positive_loss + memory_feature_negative_loss

        return feature_feature_circle_loss, memory_feature_loss

    def get_sim_seperation_loss(self, input_dict):
        """This loss ensures the memory vectors are distinguishable to prevent all memory vectors from collapsing into a single class."""
        positive_thres = self.config.memory_circle_loss_m
        negative_thres = 1 - positive_thres
        memory_vector = input_dict["memory_vector"]
        memory_size = memory_vector.shape[0]
        # First calculate separation loss based on cosine similarity
        key_value_cos_sim = cos_similar(memory_vector, memory_vector)
        key_value_cos_sim -= torch.eye(memory_size).cuda()
        key_value_cos_sim = F.relu(key_value_cos_sim - negative_thres)
        key_value_cos_sim = key_value_cos_sim.mean() / 2  # Since the similarity matrix is symmetric, we only compute it once

        return key_value_cos_sim

    def get_class_constraint(self, input_dict):
        class_label = input_dict['taxonomy_ids']
        pred_class_label = input_dict["pred_class_label"]
        ce_loss = F.cross_entropy(pred_class_label, class_label)

        return ce_loss

    def get_rebuild_loss(self, input_dict, epoch=0):
        coarse_points = input_dict["coarse_points"]
        rebuild_points = input_dict["rebuild_points"]
        gt = input_dict["gt_points"]

        loss_coarse = self.loss_func(coarse_points, gt)
        loss_fine = self.loss_func(rebuild_points, gt)
        return torch.tensor(0).cuda(), loss_coarse, loss_fine

    def forward(self, input_dict):
        xyz = input_dict["partial_points"]
        bs, n, _ = xyz.shape

        adj = input_dict["adj"]
        gnn_feature = self.gnn1_embed(xyz, adj, None)
        gnn_feature = torch.max(gnn_feature, 1)[0]
        gnn_feature = self.bn_fc2(self.fc2(F.relu(self.bn_fc1(self.fc1(gnn_feature)))))
        pred_class_label = self.classifier(gnn_feature.unsqueeze(2)).squeeze(2)
        input_dict["pred_class_label"] = pred_class_label

        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))  # B 256 n
        feature = feature * (1 / torch.sqrt(torch.tensor(256)) / feature.norm(float('inf'), dim=[1], keepdim=True))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        input_dict["encoder_cls_token"] = feature_global

        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # B 512 n
        feature = self.second_conv(feature)  # B 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B 1024

        if self.training:
            # Use point encoder (DGCNN) to extract features of gt point cloud
            gt_points = input_dict["gt_points"]
            _, gt_n, _ = gt_points.shape
            gt_feature = self.first_conv(gt_points.transpose(2, 1))  # B 256 n
            gt_feature = gt_feature * (1 / torch.sqrt(torch.tensor(256)) / gt_feature.norm(float('inf'), dim=[1], keepdim=True))
            gt_feature_global = torch.max(gt_feature, dim=2, keepdim=True)[0]  # B 384
            gt_feature = torch.cat([gt_feature_global.expand(-1, -1, gt_n), gt_feature], dim=1)  # B 512 n
            gt_feature = self.second_conv(gt_feature)  # B 1024 n
            gt_feature = torch.max(gt_feature, dim=2, keepdim=False)[0]  # B 1024
            input_dict["gt_points_token"] = gt_feature

            input_dict["keys"] = self.memory_vector[:, :self.encoder_embed_dim]
            input_dict["values"] = self.memory_vector[:, -self.decoder_embed_dim:]
            input_dict["memory_vector"] = self.memory_vector
            keys_reparam = self.memory_vector[:, :self.encoder_embed_dim]
            values_reparam = self.memory_vector[:, -self.decoder_embed_dim:]
        else:
            keys_reparam = self.memory_vector[:, :self.encoder_embed_dim]
            values_reparam = self.memory_vector[:, -self.decoder_embed_dim:]

        memory_q = self.memory_query_mlp(input_dict["encoder_cls_token"]).squeeze(2)
        memory_k = self.memory_key_mlp(keys_reparam.unsqueeze(2)).squeeze(2)

        # Prevent gradient vanishing by dividing by a coefficient
        attn = (memory_q @ memory_k.T) / torch.sqrt(torch.tensor(self.memory_size))
        attn = torch.softmax(attn, dim=1)
        sigmoid_attn = torch.sigmoid(attn)
        gating_param = self.gating_alpha * sigmoid_attn + self.gating_beta
        gating_param = F.relu(torch.tanh(gating_param))
        gating_param = gating_param / (gating_param + 1e-10)

        # Extract semantic aware feature using max
        semantic_aware_feat = (gating_param.unsqueeze(2) * attn.unsqueeze(2) * values_reparam.unsqueeze(0)).max(1)[0]
        input_dict["semantic_aware_feat"] = semantic_aware_feat

        feature_global = feature_global + semantic_aware_feat

        feature_global = torch.cat((feature_global, gnn_feature), dim=1).unsqueeze(2)
        feature_global = self.feature_align(feature_global).squeeze(2)

        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.number_coarse, 3)  # B M 3
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)  # B M S 3
        point_feat = point_feat.reshape(-1, self.number_fine, 3).transpose(2, 1)  # B 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs, -1, self.number_coarse, -1)  # B 2 M S
        seed = seed.reshape(bs, -1, self.number_fine)  # B 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.number_fine)  # B 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # B C N

        fine = self.final_conv(feat) + point_feat  # B 3 N

        input_dict["coarse_points"] = coarse.contiguous()
        input_dict["rebuild_points"] = fine.transpose(1, 2).contiguous()
        input_dict["sampled_coarse"] = input_dict["rebuild_points"]  # Placeholder, no effect

        return input_dict
