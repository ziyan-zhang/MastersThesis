# -*- coding: utf-8 -*-
# @Time    : 2021/3/18 14:30
# @Author  : WC_Server
# @File    : rankDanDu.py
# @备注     : 


# -*- coding: utf-8 -*-
# @Time    : 2021/3/16 15:51
# @Author  : WC_Server
# @File    : secondModel_RankwithType.py
# @备注     :


import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
# from tensorboardX import SummaryWriter

class SecondModel(nn.Module):  # 3085185个参数
    def __init__(self):
        super(SecondModel, self).__init__()
        self.model_name = Path(__file__).name[:-3]

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),  # 64, 128  0
            nn.LeakyReLU(0.2, inplace=True),  # 1

            nn.Conv2d(32, 32, 4, 2, 1, bias=False),  # 64, 64  2
            nn.BatchNorm2d(32),  # 3
            nn.LeakyReLU(0.2, inplace=True),  # 4
        )

        self.skip1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1, bias=False),  # 128, 64  5
            nn.BatchNorm2d(64),  # 6
            nn.LeakyReLU(0.2, inplace=True),  # 7

            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),  # 256, 16  # 8
            nn.BatchNorm2d(64 * 2),  # 9
            nn.LeakyReLU(0.2, inplace=True),  # 10
        )

        self.skip2 = nn.Sequential(
            nn.Conv2d(64, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64*4, 64*8, 4, 2, 1, bias=False),  # 512, 8  11
            nn.BatchNorm2d(64*8),  # 12
            nn.LeakyReLU(0.2, inplace=True),  #13
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_classify1 = nn.Linear(512, 16)
        self.dp_classify1 = nn.Dropout(0.5)
        self.fc_classify2 = nn.Linear(16, 5)

        self.fc_rank1 = nn.Linear(512, 16)
        self.dp_rank1 = nn.Dropout(0.5)
        self.fc_rank2 = nn.Linear(16, 3)

        self.fc_score1 = nn.Linear(512, 16)
        self.dp_score1 = nn.Dropout(0.5)
        self.fc_score2 = nn.Linear(16, 1)

    def forward(self, input):
        block1_out = self.block1(input)
        skip1_out = self.skip1(input)
        block2_in = torch.cat((block1_out, skip1_out), 1)

        block2_out = self.block2(block2_in)
        skip2_out = self.skip2(block2_in)

        block3_in = torch.cat((block2_out, skip2_out), 1)
        block3_out = self.block3(block3_in)

        output = self.avgpool(block3_out)
        output = torch.flatten(output, 1)

        output_classify1 = self.fc_classify1(output)
        output_classify1 = self.dp_classify1(output_classify1)
        output_classify2 = self.fc_classify2(output_classify1)

        output_rank1 = self.fc_rank1(output)
        output_rank1 = self.dp_rank1(output_rank1)
        output_rank2 = self.fc_rank2(output_rank1)

        output_score1 = self.fc_score1(output)
        output_score1 = self.dp_score1(output_score1)
        output_score2 = self.fc_score2(output_score1)

        return output_classify2, output_rank2, output_score2


if __name__ == '__main__':
    model = SecondModel().cuda()
    inputs = torch.randn(13, 1, 256, 256).cuda()
    outputs = model(inputs)

    # torchviz方式. inputs后面跟.requires_grad_(True)显示x输入形状框
    from torchviz import make_dot
    vis_graph = make_dot(model(inputs.requires_grad_(True)),
                         params=dict(list(model.named_parameters()) + [('x', inputs)]))
    vis_graph.view()

    # TensorboardX方式
    with SummaryWriter(comment='vgg') as w:
        w.add_graph(model, (inputs,))