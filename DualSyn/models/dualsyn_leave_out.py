import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
import pandas as pd
import numpy as np
import argparse
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor
from models.layer import GNNLayer


parser = argparse.ArgumentParser(description='Process some floats.')
parser.add_argument('--leave_type', type=str, help='The type of leaveout')
parser.add_argument('--dropping_method', type=str, help='The type of drop')
parser.add_argument('--dropout_rate', type=float, help='The dropout rate')
parser.add_argument('--device_num', type=int, help='The number of device')

args = parser.parse_args()
            
dropping_method = args.dropping_method
dropout_rate = args.dropout_rate
device_num = args.device_num


class DNN(nn.Module):
    
    def __init__(self, layers, dropout=0.2):
        super(DNN, self).__init__()

        self.dnn_network = nn.ModuleList([
            nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))
            ]+ [
            nn.BatchNorm1d(layer[1]) for layer in list(zip(layers[:-1], layers[1:]))
        ])
        self.dropout = nn.Dropout(p=dropout)

        self.activate = nn.LeakyReLU()

    def forward(self, x):

        step = int(len(self.dnn_network)/2)
        for i in range(step):
            linear = self.dnn_network[i]
            batchnorm = self.dnn_network[i + step]
            
            x = self.dropout(x)
            x = linear(x)
            x = batchnorm(x)
            x = self.activate(x)
       
        x = self.dropout(x)
        
        return x

class ProductLayer(nn.Module):
    
    def __init__(self, mode, embed_dim, field_num, hidden_units):
        
        super(ProductLayer, self).__init__()
        self.mode = mode
        self.w_z = nn.Parameter(torch.rand([field_num, embed_dim, hidden_units[0]])) 

        if mode == 'in':
            self.w_p = nn.Parameter(torch.rand([field_num, field_num, hidden_units[0]]))
        else:
            self.w_p = nn.Parameter(torch.rand([embed_dim, embed_dim, hidden_units[0]]))

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU()  
        )

    
    def forward(self, z, sparse_embeds):  
        device = torch.device('cuda:'+str(device_num))
        l_z = z.view(z.size()[0], -1)
        l_z = F.normalize(l_z, 2, 1)
        l_z = self.fc(l_z)
        
        if self.mode == 'in':  
            p = torch.matmul(sparse_embeds, sparse_embeds.permute((0, 2, 1)))
        else:  
            f_sum = torch.unsqueeze(torch.sum(sparse_embeds, dim=1), dim=1)  
            p = torch.matmul(f_sum.permute((0, 2,1)), f_sum)     
        
        l_p = torch.mm(p.reshape(p.shape[0], -1), self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T) 

        output = l_p.to(device) + l_z.to(device)

        return output


class DualSyn(torch.nn.Module):
    def __init__(self, n_output = 1, n_filters=32, embed_dim=128, num_features_xd=64, num_features_xt=954, output_dim=128, dropout=0.1):

        super(DualSyn, self).__init__()

        self.activate = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # SMILES1 graph branch
        self.n_output = n_output 
        #self.drug_conv1 = TransformerConv(78, num_features_xd * 2,  heads = 2)
        self.drug_conv1 = GNNLayer(78, num_features_xd * 2,  heads = 2, dropping_method = dropping_method)
        self.drug_ln1 = nn.LayerNorm(num_features_xd * 4)
        #self.drug_conv2 = TransformerConv(num_features_xd * 4, num_features_xd * 8, heads = 2)
        self.drug_conv2 = GNNLayer(num_features_xd * 4, num_features_xd * 8, heads = 2, dropping_method = dropping_method)
        self.drug_ln2 = nn.LayerNorm(num_features_xd * 16)
        
        self.drug_fc_g1 = torch.nn.Linear(num_features_xd * 16, num_features_xd * 8)
        self.drug_ln3 = nn.LayerNorm(num_features_xd * 8)
        self.drug_fc_g2 = torch.nn.Linear(num_features_xd * 8, output_dim )
        self.drug_ln4 = nn.LayerNorm(output_dim)


        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
       
        mode = 'in'
        field_num = 3  
        hidden_units = [256, 64]
        dnn_dropout = 0.2

        self.product = ProductLayer(mode, embed_dim, field_num, hidden_units)
        
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)


    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1 = self.drug_conv1(x1, edge_index1, dropout_rate)
        x1 = self.activate(x1)

        x1 = self.drug_conv2(x1, edge_index1, dropout_rate)
        x1 = self.activate(x1)
                
        x1 = gmp(x1, batch1)       # global max pooling

        # flatten
        x1 = self.drug_fc_g1(x1)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)

        x1 = self.drug_fc_g2(x1)
        x1 = self.dropout(x1)


        # deal drug2
        x2 = self.drug_conv1(x2, edge_index2, dropout_rate)
        x2 = self.activate(x2)

        x2 = self.drug_conv2(x2, edge_index2, dropout_rate)
        x2 = self.activate(x2)

        x2= gmp(x2, batch2)  # global max pooling


        # flatten
        x2 = self.drug_fc_g1(x2)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)
        
        x2 = self.drug_fc_g2(x2)
        x2 = self.dropout(x2)

        # deal cell
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)
        sparse_embeds = torch.stack([x1, x2, cell_vector], dim=1)   
        z = sparse_embeds

        # product layer
        sparse_inputs = self.product(z, sparse_embeds)
        
        
        # dnn_network
        dnn_x = self.dnn_network(sparse_inputs)
        final = self.dense_final(dnn_x)

        outputs = torch.sigmoid(final.squeeze(1))
        
        return outputs 



