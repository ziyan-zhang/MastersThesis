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

        self.fc_classify = nn.Linear(512, 16)
        self.dp_classify = nn.Dropout(0.5)

        self.fc_classify_rank = nn.Linear(16, 3)
        self.fc_classify_type = nn.Linear(16, 5)

        self.fc_score = nn.Linear(512, 16)
        self.dp_score = nn.Dropout(0.5)
        self.fc_score_value = nn.Linear(16, 1)

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

        output_classify = self.fc_classify(output)
        output_classify = self.dp_classify(output_classify)
        output_classify_type = self.fc_classify_type(output_classify)
        output_classify_rank = self.fc_classify_rank(output_classify)

        output_score = self.fc_score(output)
        output_score = self.dp_score(output_score)
        output_score_value = self.fc_score_value(output_score)

        return output_classify_type, output_classify_rank, output_score_value


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