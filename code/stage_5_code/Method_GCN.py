
'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.method import method
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GraphConvolution import GraphConvolution
from code.base_class.evaluate import evaluate
from code.stage_5_code.Evaluate_Accuracy import  Evaluate_Accuracy
import scipy.sparse as sp
import numpy as np
import time
import random
import math
from torch.nn.parameter import Parameter

class MethodGCN(nn.Module):
    data = None
    train_test_val = None
    lr = 0.01
    weight_decay = 5e-4
    epoch = 200

    def __init__(self, nfeat, nhid, nclass, dropout, seed):
        nn.Module.__init__(self)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, raw_x, adj, eigen_adj=None):
        x = F.relu(self.gc1(raw_x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        pred_y = F.log_softmax(x, dim=1)
        return pred_y

    def train_model(self, epoch_iter):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        accuracy = Evaluate_Accuracy('', '')

        features = self.data['graph']
        features = features['X']
        print(features)
        adj = self.data['graph']
        adj = adj['utility']
        adj = adj['A']
        self.train_test_val = self.data['train_test_val']
        self.data = self.data['graph']

        for epoch in range(epoch_iter):
            t_epoch_begin = time.time()
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, adj)
            loss_train = F.nll_loss(output[self.train_test_val['idx_train']], self.data['y'][self.train_test_val['idx_train']])
            accuracy.data = {'true_y': self.data['y'][self.train_test_val['idx_train']], 'pred_y': output[self.train_test_val['idx_train']].max(1)[1]}
            acc_train = accuracy.evaluate()
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.data['X'], adj)

            loss_val = F.nll_loss(output[self.train_test_val['idx_val']], self.data['y'][self.train_test_val['idx_val']])
            accuracy.data = {'true_y': self.data['y'][self.train_test_val['idx_val']], 'pred_y': output[self.train_test_val['idx_val']].max(1)[1]}
            acc_val = accuracy.evaluate()

            loss_test = F.nll_loss(output[self.train_test_val['idx_test']], self.data['y'][self.train_test_val['idx_test']])
            accuracy.data = {'true_y': self.data['y'][self.train_test_val['idx_test']],
                             'pred_y': output[self.train_test_val['idx_test']].max(1)[1]}
            acc_test = accuracy.evaluate()
            if epoch%10 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'loss_test: {:.4f}'.format(loss_test.item()),
                      'acc_test: {:.4f}'.format(acc_test.item()),
                      'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    def test_model(self):
        self.eval()

        adj = self.data['utility']
        adj = adj['A']

        accuracy = Evaluate_Accuracy()
        output = self.forward(self.data['X'], adj)
        loss_test = F.nll_loss(output[self.train_test_val['idx_test']], self.data['y'][self.train_test_val['idx_test']])
        accuracy.data = {'true_y': self.data['y'][self.train_test_val['idx_test']], 'pred_y': output[self.train_test_val['idx_test']].max(1)[1]}
        acc_test = accuracy.evaluate()
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return {'true_y': self.data['y'][self.train_test_val['idx_test']], 'pred_y': output[self.train_test_val['idx_test']].max(1)[1]}, acc_test.item()

    def run(self):
        self.train_model(self.epoch)
        result, test_acc = self.test_model()
        return result, test_acc
