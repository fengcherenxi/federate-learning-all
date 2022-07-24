# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 12:25
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
import copy
from itertools import chain

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from tqdm import tqdm

from get_data import nn_seq_wind


def get_val_loss(args, model, Val):
    # 返回测试集的loss
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(args, model, server):
    model.train()
    # 获取到训练验证测试
    Dtr, Val, Dte = nn_seq_wind(model.name, args.B)
    # train的长度
    model.len = len(Dtr)
    # 从服务器上获取global_model
    global_model = copy.deepcopy(server)
    lr = args.lr
    # 初始化优化器
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    # 进行固定步长衰减学习率
    stepLR = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    print('training...')
    # 定义loss的计算方式
    loss_function = nn.MSELoss().to(args.device)
    # E本地训练轮数
    for epoch in tqdm(range(args.E)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            optimizer.zero_grad()
            # compute proximal_term
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                # 全局模型和本地模型之间的差值
                proximal_term += (w - w_t).norm(2)
            # 正则化loss项
            loss = loss_function(y_pred, label) + (args.mu / 2) * proximal_term
            # 训练loss添加loss
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        # 进行概率衰减
        stepLR.step()
        # validation
        # 返回验证集的loss
        val_loss = get_val_loss(args, model, Val)
        # 调整更新min_val_loss
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()
    return best_model

def test(args, ann):
    ann.eval()
    Dtr, Val, Dte = nn_seq_wind(ann.name, args.B)
    pred = []
    y = []
    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = ann(seq)
            # 直接扩展两个列表
            # chain.from_iterable将多个迭代器连接起来
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))

    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:', np.sqrt(mean_squared_error(y, pred)))