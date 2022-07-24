# -*- coding:utf-8 -*-
"""
@Time: 2022/03/03 13:01
@Author: KI
@File: main.py
@Motto: Hungry And Humble
"""
from args import args_parser
from server import FedProx


def main():
    args = args_parser()
    fedProx = FedProx(args)
    # 全局训练
    fedProx.server()
    # 进行测试
    fedProx.global_test()


if __name__ == '__main__':
    main()