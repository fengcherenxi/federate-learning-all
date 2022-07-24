
# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:50
@Author: KI
@File: server.py
@Motto: Hungry And Humble
"""
import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from model import ANN
from client import train, test


class FedProx:
    def __init__(self, args):
        self.args = args
        # 初始化服务器模型
        self.nn = ANN(args=self.args, name='server').to(args.device)
        # 初始化参与的客户端模型
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        # r是服务端和客户端的通信轮数
        for t in tqdm(range(self.args.r)):
            print('round', t + 1, ':')
            # 客户端进行采样
            m = np.max([int(self.args.C * self.args.K), 1])
            # 0-k进行抽取m个号，形成一个数组
            index = random.sample(range(0, self.args.K), m)  # st
            # dispatch将全局聚合的模型下发到本地local model
            self.dispatch(index)
            # local updating本地模型进行更新
            self.client_update(index)
            # aggregation聚合本地模型
            self.aggregation(index)
        # 返回聚合后的模型
        return self.nn

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():# 产生模块的名称以及模块本身
            params[k] = torch.zeros_like(v.data)# 产生大小一样的全零张量

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()

    def dispatch(self, index):
        for j in index:
            # 获取到上一轮对应id的参数以及G_model参数
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()# 将全局模型下发到对应id的局部模型

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.args, self.nns[k], self.nn)

    def global_test(self):
        model = self.nn
        model.eval()
        for client in self.args.clients:
            model.name = client
            test(self.args, model)