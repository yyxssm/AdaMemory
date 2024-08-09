import os.path as osp
import os
from math import ceil

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"    # debug用，记得删了
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module,BatchNorm1d, Linear, Conv1d
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

# from models.src.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.chamfer_dist import ChamferDistance
# from src.chamfer_distance.chamfer_distance import ChamferDistance

class GNN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = BatchNorm1d(out_channels)

        if lin is True:
            self.lin = Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)   # 先从BxNxC->BNxC, norm之后再view回去？
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()

        x0 = x
        # x1 = self.bn(1, self.conv1(x0, adj, mask).relu())  # 注意这里是先relu再BN
        # x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        # x3 = self.bn(3, self.conv3(x2, adj, mask).relu())
        x1 = self.bn(1, self.conv1(x0, adj, mask)).relu()  # 暂时改为BN再ReLU
        x2 = self.bn(2, self.conv2(x1, adj, mask)).relu()
        x3 = self.bn(3, self.conv3(x2, adj, mask)).relu()
        
        x = torch.cat([x1, x2, x3], dim=-1)
        # x = x3    # 不使用residual结构

        if self.lin is not None:
            x = self.lin(x).relu()

        return x

# PyG
class diffpool_SAGE(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, output_dim=3,hidden_dim=128, embedding_dim=128,
        label_dim=40,linkpred=False):
        super().__init__()
        self.max_num_nodes=max_num_nodes
        self.linkpred = linkpred

        num_nodes = ceil(0.25 * max_num_nodes)
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_nodes)
        self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(2 * hidden_dim + embedding_dim, hidden_dim, num_nodes)
        self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, hidden_dim, embedding_dim, lin=False)

        # 1024 x 0.25 x 0.25 x 0.5 = 32
        num_nodes = ceil(0.5 * num_nodes)
        self.gnn3_pool = GNN(2 * hidden_dim + embedding_dim, 64, num_nodes)
        self.gnn3_embed = GNN(2 * hidden_dim + embedding_dim, 64, 64, lin=False)

        self.gnn4_embed = GNN(3 * 64, 64, 64, lin=False)
        self.lin_layer = Linear(3 * 64, output_dim)
        # 不做分类，只采样，分类交给 PointNet
        # self.lin1 = Linear(3 * 64, 64)
        # self.lin2 = Linear(64, label_dim)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        s = self.gnn3_pool(x, adj)
        x = self.gnn3_embed(x, adj)

        x, adj, l3, e3 = dense_diff_pool(x, adj, s)

        x = self.gnn4_embed(x, adj)  # Bx32x192
        x = self.lin_layer(x)

        # x = x.mean(dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3
        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
            # ref_pc and samp_pc are B x N x 3 matrices
            cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
            max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
            max_cost = torch.mean(max_cost)
            cost_p1_p2 = torch.mean(cost_p1_p2)
            cost_p2_p1 = torch.mean(cost_p2_p1)
            loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
            return loss

    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                # print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            # return loss + self.link_loss
            return self.link_loss

        return torch.tensor(0).to(adj)
        # return loss

# 只用一次pool
class diffpool_SAGE_onepool(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, output_dim=3,hidden_dim=128, embedding_dim=128,
        label_dim=40, assign_ratio=1/32, out_normalize_dim=1, linkpred=False):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.out_normalize_dim = out_normalize_dim
        self.linkpred = linkpred

        num_nodes = ceil(assign_ratio * max_num_nodes)
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_nodes)
        self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)

        # self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, 64, 64, lin=False)
        # self.lin_layer = Linear(3 * 64, output_dim)
        self.conv1 = torch.nn.Conv1d(2 * hidden_dim + embedding_dim,256,1)
        self.conv2 = torch.nn.Conv1d(256,128,1)
        self.conv3 = torch.nn.Conv1d(128,64,1)
        self.conv4 = torch.nn.Conv1d(64,3,1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # 不做分类，只采样，分类交给 PointNet
        # self.lin1 = Linear(3 * 64, 64)
        # self.lin2 = Linear(64, label_dim)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        # x = self.gnn2_embed(x, adj)  # Bx32x192
        # x = self.lin_layer(x)
        if self.out_normalize_dim:
            x = F.normalize(x,p=2,dim=self.out_normalize_dim)

        # finalmlp
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(1,2)

        # x = x.mean(dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3
        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
            # ref_pc and samp_pc are B x N x 3 matrices
            cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
            max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
            max_cost = torch.mean(max_cost)
            cost_p1_p2 = torch.mean(cost_p1_p2)
            cost_p2_p1 = torch.mean(cost_p2_p1)
            loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
            return loss

    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                # print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            # return loss + self.link_loss
            return self.link_loss

        return torch.tensor(0).to(adj)
        # return loss

class gcnpooling_SAGE(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, output_dim=3,hidden_dim=128, embedding_dim=128,
        label_dim=40, assign_ratio=1/32,linkpred=False):
        super().__init__()
        self.max_num_nodes=max_num_nodes
        self.linkpred = linkpred

        self.num_out_points = ceil(assign_ratio * max_num_nodes)
        self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)

        # self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, 64, 64, lin=False)
        # self.lin_layer = Linear(3 * 64, output_dim)
        self.fc1 = nn.Linear(2 * hidden_dim + embedding_dim, 256)  # in:384 out:256
        # self.fc1 = nn.Linear(128, 256)    # 当不使用residual时使用，消融实验
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * self.num_out_points)   # 3x32

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        # 不做分类，只采样，分类交给 PointNet
        # self.lin1 = Linear(3 * 64, 64)
        # self.lin2 = Linear(64, label_dim)

    def forward(self, x, adj, mask=None):
        x = self.gnn1_embed(x, adj, mask)
        x = torch.max(x, 1)[0]

        # finalmlp
        x = F.relu(self.bn_fc1(self.fc1(x))) # 256
        x = F.relu(self.bn_fc2(self.fc2(x))) # 256
        x = F.relu(self.bn_fc3(self.fc3(x))) # 256
        x = self.fc4(x) # B,96

        x = x.view(-1,self.num_out_points, 3) # B,32,3
        x = x.contiguous()

        # x = x.mean(dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3
        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss
    def get_simplification_loss_4iterm(self, ref_pc, samp_pc, beta=1,gamma=1):
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        max_cost2 = torch.max(cost_p2_p1, dim=1)[0]  # furthest point
        max_cost2 = torch.mean(max_cost2)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = beta * (cost_p1_p2 + cost_p2_p1) + gamma * (max_cost + max_cost2)
        return loss
    def get_simplification_loss_2iterm(self, ref_pc, samp_pc, beta=1,gamma=1):
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, _ = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        loss = beta * cost_p1_p2 + gamma * max_cost 
        return loss
    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                # print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            # return loss + self.link_loss
            return self.link_loss

        return torch.tensor(0).to(adj)


class diffpool_gcnpooling_fusion(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, output_dim=3,hidden_dim=128, embedding_dim=128,
        label_dim=40, assign_ratio=1/32, out_normalize_dim=1,linkpred=True):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.out_normalize_dim = out_normalize_dim
        self.linkpred = linkpred
        num_nodes = ceil(assign_ratio * max_num_nodes)
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_nodes)
        self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)

        # self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, 64, 64, lin=False)
        # self.lin_layer = Linear(3 * 64, output_dim)
        self.conv1 = torch.nn.Conv1d(2*(2 * hidden_dim + embedding_dim),256,1)
        self.conv2 = torch.nn.Conv1d(256,128,1)
        self.conv3 = torch.nn.Conv1d(128,64,1)
        self.conv4 = torch.nn.Conv1d(64,3,1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # 不做分类，只采样，分类交给 PointNet
        # self.lin1 = Linear(3 * 64, 64)
        # self.lin2 = Linear(64, label_dim)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        # self.assign_tensor = torch.softmax(s, dim=-1)
        
        aggx = torch.max(x, 1)[0]  # B,384
        B,C = aggx.shape
        aggx = aggx.reshape(B,1,C) # B,1,384
        aggx = aggx.expand(B,s.shape[2],C)  # B,32,384


        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask) # B,32,384   l1值很小只有10的-5次方大小

        self.l1 = l1
        self.e1 = e1

        # x = self.gnn2_embed(x, adj)  # Bx32x192
        # x = self.lin_layer(x)
        if self.out_normalize_dim:
            x = F.normalize(x,p=2,dim=self.out_normalize_dim)

        x = torch.cat([x, aggx], dim=-1)  # B,32,384*2


        # finalmlp
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(1,2)

        # x = x.mean(dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3
        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
            # ref_pc and samp_pc are B x N x 3 matrices
            cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
            max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
            max_cost = torch.mean(max_cost)
            cost_p1_p2 = torch.mean(cost_p1_p2)
            cost_p2_p1 = torch.mean(cost_p2_p1)
            loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
            return loss

    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        # eps = 1e-7
        # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        # if self.linkpred:
        #     max_num_nodes = adj.size()[1]
        #     pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
        #     tmp = pred_adj0
        #     pred_adj = pred_adj0
        #     for adj_pow in range(adj_hop-1):
        #         tmp = tmp @ pred_adj0
        #         pred_adj = pred_adj + tmp
        #     pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
        #     #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
        #     #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
        #     #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
        #     self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
        #     if batch_num_nodes is None:
        #         num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        #         # print('Warning: calculating link pred loss without masking')
        #     else:
        #         num_entries = np.sum(batch_num_nodes * batch_num_nodes)
        #         embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        #         adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
        #         self.link_loss[(1-adj_mask).bool()] = 0.0

        #     self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        #     #print('linkloss: ', self.link_loss)
        #     # return loss + self.link_loss
        #     return self.link_loss
        if self.linkpred:
            return self.l1 + self.e1

        return torch.tensor(0).to(adj)
        # return loss


class diffpool_gcnpooling_fusion_upsample(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, output_dim=3,hidden_dim=128, embedding_dim=128,
        label_dim=40, assign_ratio=1/32, out_normalize_dim=1,linkpred=True):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.out_normalize_dim = out_normalize_dim
        self.linkpred = linkpred
        num_nodes = ceil(assign_ratio * max_num_nodes)
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_nodes)
        self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)

        # self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, 64, 64, lin=False)
        # self.lin_layer = Linear(3 * 64, output_dim)
        self.conv1 = torch.nn.Conv1d(2*(2 * hidden_dim + embedding_dim),256,1)
        self.conv2 = torch.nn.Conv1d(256,128,1)
        self.conv3 = torch.nn.Conv1d(128,64,1)
        self.conv4 = torch.nn.Conv1d(64,3,1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # 不做分类，只采样，分类交给 PointNet
        # self.lin1 = Linear(3 * 64, 64)
        # self.lin2 = Linear(64, label_dim)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        # self.assign_tensor = torch.softmax(s, dim=-1)
        
        aggx = torch.max(x, 1)[0]  # B,384
        B,C = aggx.shape
        aggx = aggx.reshape(B,1,C) # B,1,384
        aggx = aggx.expand(B,s.shape[2],C)  # B,32,384


        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask) # B,32,384   l1值很小只有10的-5次方大小

        self.l1 = l1
        self.e1 = e1

        # x = self.gnn2_embed(x, adj)  # Bx32x192
        # x = self.lin_layer(x)
        if self.out_normalize_dim:
            x = F.normalize(x,p=2,dim=self.out_normalize_dim)

        x = torch.cat([x, aggx], dim=-1)  # B,32,384*2


        # finalmlp
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # 生成的点
        x = x.transpose(1,2)



        # x = x.mean(dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3
        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
            # ref_pc and samp_pc are B x N x 3 matrices
            cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
            max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
            max_cost = torch.mean(max_cost)
            cost_p1_p2 = torch.mean(cost_p1_p2)
            cost_p2_p1 = torch.mean(cost_p2_p1)
            loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
            return loss

    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        # eps = 1e-7
        # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        # if self.linkpred:
        #     max_num_nodes = adj.size()[1]
        #     pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
        #     tmp = pred_adj0
        #     pred_adj = pred_adj0
        #     for adj_pow in range(adj_hop-1):
        #         tmp = tmp @ pred_adj0
        #         pred_adj = pred_adj + tmp
        #     pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
        #     #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
        #     #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
        #     #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
        #     self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
        #     if batch_num_nodes is None:
        #         num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        #         # print('Warning: calculating link pred loss without masking')
        #     else:
        #         num_entries = np.sum(batch_num_nodes * batch_num_nodes)
        #         embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        #         adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
        #         self.link_loss[(1-adj_mask).bool()] = 0.0

        #     self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        #     #print('linkloss: ', self.link_loss)
        #     # return loss + self.link_loss
        #     return self.link_loss
        if self.linkpred:
            return self.l1 + self.e1

        return torch.tensor(0).to(adj)
        # return loss



class diffpool_gcnpooling_fusion_GAafterGE(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, output_dim=3,hidden_dim=128, embedding_dim=128,
        label_dim=40, assign_ratio=1/32, out_normalize_dim=1):
        super().__init__()
        self.max_num_nodes = max_num_nodes
        self.out_normalize_dim = out_normalize_dim
        num_nodes = ceil(assign_ratio * max_num_nodes)
        
        self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)
        self.gnn1_pool = GNN(2 * hidden_dim + embedding_dim, hidden_dim, num_nodes)

        # self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, 64, 64, lin=False)
        # self.lin_layer = Linear(3 * 64, output_dim)
        self.conv1 = torch.nn.Conv1d(2*(2 * hidden_dim + embedding_dim),256,1)
        self.conv2 = torch.nn.Conv1d(256,128,1)
        self.conv3 = torch.nn.Conv1d(128,64,1)
        self.conv4 = torch.nn.Conv1d(64,3,1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        # 不做分类，只采样，分类交给 PointNet
        # self.lin1 = Linear(3 * 64, 64)
        # self.lin2 = Linear(64, label_dim)

    def forward(self, x, adj, mask=None):
        x = self.gnn1_embed(x, adj, mask)
        s = self.gnn1_pool(x, adj, mask)

        aggx = torch.max(x, 1)[0]  # B,384
        B,C = aggx.shape
        aggx = aggx.reshape(B,1,C) # B,1,384
        aggx = aggx.expand(B,s.shape[2],C)  # B,32,384


        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask) # B,32,384

        # x = self.gnn2_embed(x, adj)  # Bx32x192
        # x = self.lin_layer(x)
        if self.out_normalize_dim:
            x = F.normalize(x,p=2,dim=self.out_normalize_dim)

        x = torch.cat([x, aggx], dim=-1)  # B,32,384*2


        # finalmlp
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(1,2)

        # x = x.mean(dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3
        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
            # ref_pc and samp_pc are B x N x 3 matrices
            cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
            max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
            max_cost = torch.mean(max_cost)
            cost_p1_p2 = torch.mean(cost_p1_p2)
            cost_p2_p1 = torch.mean(cost_p2_p1)
            loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
            return loss

    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        # eps = 1e-7
        # # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        # if self.linkpred:
        #     max_num_nodes = adj.size()[1]
        #     pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
        #     tmp = pred_adj0
        #     pred_adj = pred_adj0
        #     for adj_pow in range(adj_hop-1):
        #         tmp = tmp @ pred_adj0
        #         pred_adj = pred_adj + tmp
        #     pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
        #     #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
        #     #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
        #     #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
        #     self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
        #     if batch_num_nodes is None:
        #         num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        #         # print('Warning: calculating link pred loss without masking')
        #     else:
        #         num_entries = np.sum(batch_num_nodes * batch_num_nodes)
        #         embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        #         adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
        #         self.link_loss[(1-adj_mask).bool()] = 0.0

        #     self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        #     #print('linkloss: ', self.link_loss)
        #     # return loss + self.link_loss
        #     return self.link_loss

        return torch.tensor(0).to(adj)
        # return loss

'''*************************补充消融实验, 多层SAGE***********************'''
class GNN_1(Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True, n=1):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = BatchNorm1d(hidden_channels)

        if lin is True:
            self.lin = Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)   # 先从BxNxC->BNxC, norm之后再view回去？
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        # x1 = self.bn(1, self.conv1(x0, adj, mask).relu())  # 注意这里是先relu再BN
        # x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        # x3 = self.bn(3, self.conv3(x2, adj, mask).relu())
        x1 = self.bn(1, self.conv1(x0, adj, mask)).relu()  # 暂时改为BN再ReLU
        
        x = torch.cat([x1], dim=-1)
        # x = x3    # 不使用residual结构

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class GNN_7(Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True, n=7):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.conv4 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.conv5 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.conv6 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn6 = BatchNorm1d(hidden_channels)
        self.conv7 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn7 = BatchNorm1d(out_channels)

        # if lin is True:
        #     self.lin = Linear(2 * hidden_channels + out_channels,
        #                                out_channels)
        # else:
        self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)   # 先从BxNxC->BNxC, norm之后再view回去？
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        # x1 = self.bn(1, self.conv1(x0, adj, mask).relu())  # 注意这里是先relu再BN
        # x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        # x3 = self.bn(3, self.conv3(x2, adj, mask).relu())
        x1 = self.bn(1, self.conv1(x0, adj, mask)).relu()  # 暂时改为BN再ReLU
        x2 = self.bn(2, self.conv2(x1, adj, mask)).relu()
        x3 = self.bn(3, self.conv3(x2, adj, mask)).relu()
        x4 = self.bn(4, self.conv4(x3, adj, mask)).relu()
        x5 = self.bn(5, self.conv5(x4, adj, mask)).relu()
        x6 = self.bn(6, self.conv6(x5, adj, mask)).relu()
        x7 = self.bn(7, self.conv7(x6, adj, mask)).relu()

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=-1)
        # x = x3    # 不使用residual结构

        return x


class GNN_15(Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True, n=15):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.conv4 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.conv5 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.conv6 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn6 = BatchNorm1d(hidden_channels)

        self.conv7 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn7 = BatchNorm1d(hidden_channels)
        self.conv8 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn8 = BatchNorm1d(hidden_channels)
        self.conv9 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn9 = BatchNorm1d(hidden_channels)
        self.conv10 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn10 = BatchNorm1d(hidden_channels)
        self.conv11 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn11 = BatchNorm1d(hidden_channels)
        self.conv12 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn12 = BatchNorm1d(hidden_channels)
        self.conv13 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn13 = BatchNorm1d(hidden_channels)
        self.conv14 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn14 = BatchNorm1d(hidden_channels)
        self.conv15 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn15 = BatchNorm1d(out_channels)

        # if lin is True:
        #     self.lin = Linear(2 * hidden_channels + out_channels,
        #                                out_channels)
        # else:
        self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)   # 先从BxNxC->BNxC, norm之后再view回去？
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        # x1 = self.bn(1, self.conv1(x0, adj, mask).relu())  # 注意这里是先relu再BN
        # x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        # x3 = self.bn(3, self.conv3(x2, adj, mask).relu())
        x1 = self.bn(1, self.conv1(x0, adj, mask)).relu()  # 暂时改为BN再ReLU
        x2 = self.bn(2, self.conv2(x1, adj, mask)).relu()
        x3 = self.bn(3, self.conv3(x2, adj, mask)).relu()
        x4 = self.bn(4, self.conv4(x3, adj, mask)).relu()
        x5 = self.bn(5, self.conv5(x4, adj, mask)).relu()
        x6 = self.bn(6, self.conv6(x5, adj, mask)).relu()
        x7 = self.bn(7, self.conv7(x6, adj, mask)).relu()

        x8 = self.bn(8, self.conv8(x7, adj, mask)).relu()
        x9 = self.bn(9, self.conv9(x8, adj, mask)).relu()
        x10 = self.bn(10, self.conv10(x9, adj, mask)).relu()
        x11 = self.bn(11, self.conv11(x10, adj, mask)).relu()
        x12 = self.bn(12, self.conv12(x11, adj, mask)).relu()
        x13 = self.bn(13, self.conv13(x12, adj, mask)).relu()
        x14 = self.bn(14, self.conv14(x13, adj, mask)).relu()
        x15 = self.bn(15, self.conv15(x14, adj, mask)).relu()

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15], dim=-1)
        # x = x3    # 不使用residual结构

        return x


class GNN_31(Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True, n=31):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.conv4 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.conv5 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn5 = BatchNorm1d(hidden_channels)
        self.conv6 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn6 = BatchNorm1d(hidden_channels)

        self.conv7 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn7 = BatchNorm1d(hidden_channels)
        self.conv8 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn8 = BatchNorm1d(hidden_channels)
        self.conv9 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn9 = BatchNorm1d(hidden_channels)
        self.conv10 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn10 = BatchNorm1d(hidden_channels)
        self.conv11 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn11 = BatchNorm1d(hidden_channels)
        self.conv12 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn12 = BatchNorm1d(hidden_channels)
        self.conv13 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn13 = BatchNorm1d(hidden_channels)
        self.conv14 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn14 = BatchNorm1d(hidden_channels)

        self.conv15 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn15 = BatchNorm1d(hidden_channels)
        self.conv16 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn16 = BatchNorm1d(hidden_channels)
        self.conv17 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn17 = BatchNorm1d(hidden_channels)
        self.conv18 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn18 = BatchNorm1d(hidden_channels)
        self.conv19 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn19 = BatchNorm1d(hidden_channels)
        self.conv20 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn20 = BatchNorm1d(hidden_channels)
        self.conv21 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn21 = BatchNorm1d(hidden_channels)
        self.conv22 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn22 = BatchNorm1d(hidden_channels)

        self.conv23 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn23 = BatchNorm1d(hidden_channels)
        self.conv24 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn24 = BatchNorm1d(hidden_channels)
        self.conv25 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn25 = BatchNorm1d(hidden_channels)
        self.conv26 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn26 = BatchNorm1d(hidden_channels)
        self.conv27 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn27 = BatchNorm1d(hidden_channels)
        self.conv28 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn28 = BatchNorm1d(hidden_channels)
        self.conv29 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn29 = BatchNorm1d(hidden_channels)
        self.conv30 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn30 = BatchNorm1d(hidden_channels)

        self.conv31 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn31 = BatchNorm1d(out_channels)

        # if lin is True:
        #     self.lin = Linear(2 * hidden_channels + out_channels,
        #                                out_channels)
        # else:
        self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)   # 先从BxNxC->BNxC, norm之后再view回去？
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        # x1 = self.bn(1, self.conv1(x0, adj, mask).relu())  # 注意这里是先relu再BN
        # x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        # x3 = self.bn(3, self.conv3(x2, adj, mask).relu())
        x1 = self.bn(1, self.conv1(x0, adj, mask)).relu()  # 暂时改为BN再ReLU
        x2 = self.bn(2, self.conv2(x1, adj, mask)).relu()
        x3 = self.bn(3, self.conv3(x2, adj, mask)).relu()
        x4 = self.bn(4, self.conv4(x3, adj, mask)).relu()
        x5 = self.bn(5, self.conv5(x4, adj, mask)).relu()
        x6 = self.bn(6, self.conv6(x5, adj, mask)).relu()
        x7 = self.bn(7, self.conv7(x6, adj, mask)).relu()

        x8 = self.bn(8, self.conv8(x7, adj, mask)).relu()
        x9 = self.bn(9, self.conv9(x8, adj, mask)).relu()
        x10 = self.bn(10, self.conv10(x9, adj, mask)).relu()
        x11 = self.bn(11, self.conv11(x10, adj, mask)).relu()
        x12 = self.bn(12, self.conv12(x11, adj, mask)).relu()
        x13 = self.bn(13, self.conv13(x12, adj, mask)).relu()
        x14 = self.bn(14, self.conv14(x13, adj, mask)).relu()
        x15 = self.bn(15, self.conv15(x14, adj, mask)).relu()

        x16 = self.bn(16, self.conv16(x15, adj, mask)).relu()
        x17 = self.bn(17, self.conv17(x16, adj, mask)).relu()
        x18 = self.bn(18, self.conv18(x17, adj, mask)).relu()
        x19 = self.bn(19, self.conv19(x18, adj, mask)).relu()
        x20 = self.bn(20, self.conv20(x19, adj, mask)).relu()
        x21 = self.bn(21, self.conv21(x20, adj, mask)).relu()
        x22 = self.bn(22, self.conv22(x21, adj, mask)).relu()
        x23 = self.bn(23, self.conv23(x22, adj, mask)).relu()

        x24 = self.bn(24, self.conv24(x23, adj, mask)).relu()
        x25 = self.bn(25, self.conv25(x24, adj, mask)).relu()
        x26 = self.bn(26, self.conv26(x25, adj, mask)).relu()
        x27 = self.bn(27, self.conv27(x26, adj, mask)).relu()
        x28 = self.bn(28, self.conv28(x27, adj, mask)).relu()
        x29 = self.bn(29, self.conv29(x28, adj, mask)).relu()
        x30 = self.bn(30, self.conv30(x29, adj, mask)).relu()
        x31 = self.bn(31, self.conv31(x30, adj, mask)).relu()

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,
                       x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31], dim=-1)
        # x = x3    # 不使用residual结构

        return x


class gcnpooling_NSAGE(torch.nn.Module):
    def __init__(self, max_num_nodes=1024, input_dim=3, output_dim=3,hidden_dim=128, embedding_dim=128,
        label_dim=40, assign_ratio=1/32,linkpred=False,n_sage=1):
        super().__init__()
        self.max_num_nodes=max_num_nodes
        self.linkpred = linkpred

        self.num_out_points = ceil(assign_ratio * max_num_nodes)

        if n_sage == 1:
            print("using 1 sage layer")
            self.gnn1_embed = GNN_1(input_dim, hidden_dim, embedding_dim, lin=False)
        if n_sage == 7:
            print("using 7 sage layer")
            self.gnn1_embed = GNN_7(input_dim, hidden_dim, embedding_dim, lin=False)
        if n_sage == 15:
            print("using 15 sage layer")
            self.gnn1_embed = GNN_15(input_dim, hidden_dim, embedding_dim, lin=False)
        if n_sage == 31:
            print("using 31 sage layer")
            self.gnn1_embed = GNN_31(input_dim, hidden_dim, embedding_dim, lin=False)

        # self.gnn1_embed = GNN(input_dim, hidden_dim, embedding_dim, lin=False)

        # self.gnn2_embed = GNN(2 * hidden_dim + embedding_dim, 64, 64, lin=False)
        # self.lin_layer = Linear(3 * 64, output_dim)
        self.fc1 = nn.Linear((n_sage - 1) * hidden_dim + embedding_dim, 256)  # in:384 out:256
        # self.fc1 = nn.Linear(128, 256)    # 当不使用residual时使用，消融实验
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * self.num_out_points)   # 3x32

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        # 不做分类，只采样，分类交给 PointNet
        # self.lin1 = Linear(3 * 64, 64)
        # self.lin2 = Linear(64, label_dim)

    def forward(self, x, adj, mask=None):
        x = self.gnn1_embed(x, adj, mask)
        x = torch.max(x, 1)[0]

        # finalmlp
        x = F.relu(self.bn_fc1(self.fc1(x))) # 256
        x = F.relu(self.bn_fc2(self.fc2(x))) # 256
        x = F.relu(self.bn_fc3(self.fc3(x))) # 256
        x = self.fc4(x) # B,96

        x = x.view(-1,self.num_out_points, 3) # B,32,3
        x = x.contiguous()

        # x = x.mean(dim=1)
        # x = self.lin1(x).relu()
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=-1), l1 + l2 + l3, e1 + e2 + e3
        return x

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss
    def get_simplification_loss_4iterm(self, ref_pc, samp_pc, beta=1,gamma=1):
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        max_cost2 = torch.max(cost_p2_p1, dim=1)[0]  # furthest point
        max_cost2 = torch.mean(max_cost2)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = beta * (cost_p1_p2 + cost_p2_p1) + gamma * (max_cost + max_cost2)
        return loss
    def get_simplification_loss_2iterm(self, ref_pc, samp_pc, beta=1,gamma=1):
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, _ = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        loss = beta * cost_p1_p2 + gamma * max_cost 
        return loss
    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        ''' 
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        # loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).cuda())
            #print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            #print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            #self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                # print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            #print('linkloss: ', self.link_loss)
            # return loss + self.link_loss
            return self.link_loss

        return torch.tensor(0).to(adj)


if __name__ == '__main__':
    from thop import profile, clever_format
    from torchsummary import summary
    hd = 128
    ed = 128
#    model = gcnpooling_SAGE(1024,3,3,hd,ed,40,1/128,False)
#    input = torch.randn(1,1024, 3)
#    adj = torch.randn(1,1024,1024)
#    flops, params = profile(model, inputs=(input, adj))
#    
#    print("FLOPs=", str(flops))
#    print("params=", str(params))
    model = gcnpooling_SAGE(1024,3,3,hd,ed,40,1/2,False).cuda()
    input = torch.randn(1,1024, 3).cuda()
    adj = torch.randn(1,1024,1024).cuda()
    flops, params = profile(model, inputs=(input, adj))
    print("FLOPs=", str(flops/1000**2)+'M')
    print("params=", str(params/1000**2)+'M')

    input_shape=(1024,3)
    input_adj = (1024,1024)
    summary(model,[input_shape,input_adj],batch_size=1)

    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))
