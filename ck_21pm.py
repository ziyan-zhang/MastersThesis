# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 18:34
# @Author  : WC_Server
# @File    : ck_21pm.py
# @备注     : 大论文第二个模型的训练代码
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"  # 必须在`import torch`语句之前设置才能生效
import torch
device = torch.device('cuda')

from progressbar import *
import time
from scipy import stats
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import torch.optim as optim
from ceutils import AverageMeter
from pathlib import Path
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from shutil import copyfile

# from datasets.DL_overlap128_stride128 import train_dataset, val_dataset, data_name, cut_number
from stdDL import train_dataset, val_dataset, data_name, cut_number

# from second_models import SecondModel
# from RankwithType import SecondModel
from rank1Qi import SecondModel


def rmse_function(a, b):
    """does not need to be array-like"""
    a = np.array(a)
    b = np.array(b)
    mse = ((a-b)**2).mean()
    rmse = math.sqrt(mse)
    return rmse


def lr_plot(lr_ori):
    return np.log10(lr_ori) * 0.27 + 1.6


def get_perform(perf_dict):
    val_srcc_mean = perf_dict['val_srcc_mean']
    val_plcc_mean = perf_dict['val_plcc_mean']
    val_rmse_mean = perf_dict['val_rmse_mean']

    val_srcc_mean = np.array(val_srcc_mean)
    theIndex = np.argmax(val_srcc_mean)

    val_srcc_mean_max = val_srcc_mean[theIndex]
    val_plcc_mean_max = val_plcc_mean[theIndex]
    val_rmse_mean_max = val_rmse_mean[theIndex]

    return val_srcc_mean_max, val_plcc_mean_max, val_rmse_mean_max


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Trainer(object):
    def __init__(self, regmodel, data_loader_train, data_loader_val, cut_number, initial_learning_rate, standard_learning_rate, train_patience, plot_epoch, save_folder):

        self.train_loader = data_loader_train
        self.val_loader = data_loader_val

        # 训练相关的参数
        self.epochs = 2000
        self.start_epoch = 0
        self.use_gpu = True
        self.counter = 0
        self.train_patience = train_patience
        self.cutNumber = cut_number  # dataname没有在里面的函数使用就不用定义类变量， cutNumber在validate类函数里面用了
        self.plot_epoch = plot_epoch
        self.saveFolder = save_folder

        # 定义模型
        self.model = regmodel

        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model)  # 就在这里wrap一下，模型就会使用所有的GPU

        self.adam_lr = initial_learning_rate
        # self.standard_learning_rate = standard_learning_rate
        print('使用的学习率为%.0e' % self.adam_lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.adam_lr)
        # 指定文件夹, 和哪一个模型状态(最新/最佳)

        self.lr_bin = []
        # 定义一些收集数据的列表
        self.train_loss_reg_bin = []
        self.train_srcc_bin = []
        self.train_plcc_bin = []

        self.val_loss_reg_bin = []
        self.val_srcc_bin = []
        self.val_plcc_bin = []

        self.val_srcc_mean_bin = []
        self.val_plcc_mean_bin = []
        self.val_rmse_mean_bin = []
        self.predict_mean_bin = []

        self.is_best_srcc_mean = True
        self.is_best_loss = True

    def train(self):
        # 第一个epoch True, 第一个epoch之后可以改
        for epoch in range(self.start_epoch, self.epochs):

            # if epoch == 50:
            #     lr = self.standard_learning_rate
            #     adjust_learning_rate(self.optimizer, lr)

            print('\nEpoch: {}/{}'.format(epoch+1, self.epochs))

            # train_loss_reg, train_srcc, train_plcc, train_klcc, train_mse = self.train_one_epoch(epoch)
            train_loss_reg, train_srcc, train_plcc = self.train_one_epoch()
            val_loss_reg, val_srcc, val_plcc, val_srcc_mean, val_plcc_mean, val_rmse_mean, predict_mean = self.validate()

            temp_lr = self.optimizer.param_groups[0]['lr']
            print('当前学习率%.1e' % temp_lr)
            self.lr_bin.append(lr_plot(temp_lr))

            f = open('%s/lr_dict.txt' % self.saveFolder, 'w')
            f.write(str(self.lr_bin))
            f.close()

            # 打印当epoch的性能
            msg1 = 'Train>>>  srcc: {0:6.4f}  plcc: {1:6.4f}   VAL>>> srcc: {2:6.4f}  plcc: {3:6.4f}        BEST_VAL>>> srcc: {4:6.4f}  plcc: {5:6.4f}'
            msg2 = '                                  VAL_MEAN>>> srcc: {0:6.4f}  plcc: {1:6.4f}   BEST_VAL_MEAN>>> srcc: {2:6.4f}  plcc: {3:6.4f}'

            if epoch >= self.start_epoch + 1:
                self.is_best_loss = (train_loss_reg < np.array(self.train_loss_reg_bin).min() - 1e-3)
                self.is_best_srcc_mean = (val_srcc_mean > np.array(self.val_srcc_mean_bin).max())

            if self.is_best_loss:
                self.counter = 0
                msg2 += '[*]'
            else:
                self.counter += 1
                print('已经%d个epoch训练损失没有下降' % self.counter)

            if self.counter > self.train_patience:
                if temp_lr > self.adam_lr * 0.01001:
                    adjust_learning_rate(self.optimizer, temp_lr*0.1)
                    self.counter = 0
                    print('这里自适应改变学习率...')
                else:
                    val_srcc_mean_max, val_plcc_mean_max, val_rmse_mean_max = get_perform(perform_dict)
                    print('>>>>>最佳   SRCC: %.4f,  PLCC: %.4f,  RMSE: %.4f   <<<<<' % (
                        val_srcc_mean_max, val_plcc_mean_max, val_rmse_mean_max))
                    print('[!]已经%d个epoch训练损失不再减小, 停止训练' % self.train_patience)

                    savefolderBig = (self.saveFolder).split('\\')[0]
                    savefolderSmall = (self.saveFolder).split('\\')[1]
                    newsavefolderSmall = 'srcc_%.4f_' % val_srcc_mean_max + savefolderSmall
                    os.rename(self.saveFolder, os.path.join(savefolderBig, newsavefolderSmall))

                    return

            self.save_checkpoint(epoch,
                                 {'epoch': epoch,
                                  'model_state': self.model.state_dict(),
                                  'optim_state': self.optimizer.state_dict(),
                                  },
                                 self.is_best_srcc_mean)

            # 记录相关数据
            self.train_loss_reg_bin.append(train_loss_reg)
            self.train_srcc_bin.append(train_srcc)
            self.train_plcc_bin.append(train_plcc)

            self.val_loss_reg_bin.append(val_loss_reg)
            self.val_srcc_bin.append(val_srcc)
            self.val_plcc_bin.append(val_plcc)
            self.val_srcc_mean_bin.append(val_srcc_mean)
            self.val_plcc_mean_bin.append(val_plcc_mean)
            self.val_rmse_mean_bin.append(val_rmse_mean)

            print(msg1.format(train_srcc, train_plcc, val_srcc, val_plcc, np.array(self.val_srcc_bin).max(), np.array(self.val_plcc_bin).max()))
            print(msg2.format(val_srcc_mean, val_plcc_mean, np.array(self.val_srcc_mean_bin).max(), np.array(self.val_plcc_mean_bin).max()))

            # 画图
            if epoch % self.plot_epoch == self.plot_epoch-1:
                plt.figure()
                plt.plot(self.train_loss_reg_bin, label='train loss_reg')
                plt.plot(self.train_srcc_bin, label='train srcc')
                # plt.plot(self.train_plcc_bin, label='train plcc')

                plt.plot(self.val_loss_reg_bin, label='val loss_reg')
                plt.plot(self.val_srcc_bin, label='val srcc')
                # plt.plot(self.val_plcc_bin, label='val plcc')

                plt.plot(self.val_srcc_mean_bin, label='val mean srcc')
                # plt.plot(self.val_plcc_mean_bin, label='val mean plcc')
                # plt.plot(self.val_rmse_mean_bin, label='val mean rmse')

                plt.plot(self.lr_bin, label='lr')

                i_ = 0
                while i_ <= epoch:
                    if i_ % 29 == 0:
                        value = self.val_srcc_mean_bin[i_]
                        plt.scatter(i_, value, marker='x')
                        plt.text(i_, value, '%.3f' % value, fontsize=15, fontproperties='Times New Roman')
                        i_ += 30
                    else:
                        i_ += 1

                plt.legend()
                plt.ylim(0, 1.1)
                plt.title('srcc_mean_最后%.4f\nsrcc_mean_最大%.4f' % (self.val_srcc_mean_bin[-1], np.array(self.val_srcc_mean_bin).max()))
                plt.savefig(os.path.join(self.saveFolder, 'perform.png'))
                plt.close('all')

            perform_dict = {'train_loss_reg': self.train_loss_reg_bin, 'train_srcc': self.train_srcc_bin, 'train_plcc': self.train_plcc_bin,
                            'val_loss_reg': self.val_loss_reg_bin, 'val_srcc': self.val_srcc_bin, 'val_plcc': self.val_plcc_bin,
                            'val_srcc_mean': self.val_srcc_mean_bin, 'val_plcc_mean': self.val_plcc_mean_bin, 'val_rmse_mean': self.val_rmse_mean_bin,
                            'predict_mean': self.predict_mean_bin}

            f = open('%s/perform_dict.txt' % self.saveFolder, 'w')
            f.write(str(perform_dict))
            f.close()

    def train_one_epoch(self):
        self.model.train()
        loss_bin = AverageMeter()
        predict_bin = []
        y_bin = []
        tic = time.time()
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', FileTransferSpeed()]
        progress = ProgressBar(widgets=widgets)
        for i, (imgRead, dist_type, dist_rank, y) in enumerate(progress(self.train_loader)):
            if self.use_gpu:
                imgRead = imgRead.to(device)
                y = y.to(device)
                dist_type = dist_type.to(device)
                dist_rank = dist_rank.to(device)
            y = y.float()  # todo: 看能不能省去

            self.batch_size = imgRead.shape[0]

            # 前向传播
            dist_type_pred, dist_rank_pred, predict = self.model(imgRead)

            predict = predict.squeeze()

            # 损失函数, 以及反向传播, 更新梯度
            loss_reg = F.l1_loss(predict, y)
            loss_type_cls = F.cross_entropy(dist_type_pred, dist_type)
            loss_rank_cls = F.cross_entropy(dist_rank_pred, dist_rank)
            # loss = loss_reg + loss_type_cls + loss_rank_cls
            # loss = loss_reg + loss_type_cls
            # loss = loss_reg
            loss = loss_reg + loss_rank_cls

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_bin.update(loss.cpu().item())

            # 收集数据
            predict_bin.extend(predict.detach().cpu().numpy())
            y_bin.extend(y.cpu().numpy())

        srcc = stats.spearmanr(predict_bin, y_bin)[0]
        plcc = stats.pearsonr(predict_bin, y_bin)[0]

        return loss_bin.avg, srcc, plcc

    def validate(self):
        self.model.eval()
        loss_bin = AverageMeter()
        predict_bin = []
        y_bin = []

        tic = time.time()
        # todo: 内存增加的话加上个with torch.no_grad()
        with torch.no_grad():

            widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                       ' ', FileTransferSpeed()]
            progress = ProgressBar(widgets=widgets)
            for i, (imgRead, dist_type, dist_rank, y) in enumerate(progress(self.val_loader)):
                if self.use_gpu:
                    imgRead= imgRead.to(device)
                    y = y.to(device)
                    dist_type = dist_type.to(device)
                    dist_rank = dist_rank.to(device)
                y = y.float()

                self.batch_size = imgRead.shape[0]

                # 更改变3-3: 前向传播模型输入改动-测试. 加上r17
                dist_type_pred, dist_rank_pred, predict = self.model(imgRead)
                predict = predict.squeeze()

                # 损失函数, 不再反向传播和更新梯度
                loss_reg = F.mse_loss(predict, y)
                loss_type_cls = F.cross_entropy(dist_type_pred, dist_type)
                loss_rank_cls = F.cross_entropy(dist_rank_pred, dist_rank)
                # loss = loss_reg + loss_type_cls + loss_rank_cls
                # loss = loss_reg + loss_type_cls
                # loss = loss_reg
                loss = loss_reg + loss_rank_cls

                loss_bin.update(loss.item())

                # 收集数据
                predict_bin.extend(predict.detach().cpu().numpy())
                y_bin.extend(y.cpu().numpy())

        predict_bin = np.array(predict_bin)
        y_bin = np.array(y_bin)

        srcc_raw = stats.spearmanr(predict_bin, y_bin)[0]
        plcc_raw = stats.pearsonr(predict_bin, y_bin)[0]

        # 求平均值
        predict_bin_rs = predict_bin.reshape(-1, self.cutNumber)
        y_bin_rs = y_bin.reshape(-1, self.cutNumber)

        predict_mean = predict_bin_rs.mean(axis=1)
        y_mean = y_bin_rs.mean(axis=1)
        assert len(y_mean) == len(self.val_loader.sampler)/self.cutNumber
        # 求CC
        srcc_mean = stats.spearmanr(predict_mean, y_mean)[0]
        plcc_mean = stats.pearsonr(predict_mean, y_mean)[0]
        rmse_mean = rmse_function(predict_mean, y_mean)

        return loss_bin.avg, srcc_raw, plcc_raw, srcc_mean, plcc_mean, rmse_mean, predict_mean

    def save_checkpoint(self, epoch, state, is_best_srcc_mean):
        filename = 'epoch' + str(epoch) + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.saveFolder, filename)
        torch.save(state, ckpt_path)

        filename_minus1 = 'epoch' + str(epoch-1) + '_ckpt.pth.tar'
        ckpt_path_minus1 = os.path.join(self.saveFolder, filename_minus1)

        if os.path.exists(ckpt_path_minus1) and ((epoch - 1) % 30 != 0):
            os.remove(ckpt_path_minus1)

        if is_best_srcc_mean:
            filename = 'model_BEST_mean.pth.tar'
            shutil.copyfile(ckpt_path, os.path.join(self.saveFolder, filename))


def main():  # 单独使用main函数是为了避免使用全局变量, 是接口更严谨
    regmodel = SecondModel()

    INITIAL_LEARNING_RATE = 1e-3
    STANDARD_LEARNING_RATE = 1e-3
    BATCH_SIZE = 256

    TRAIN_PATIENCE = 10

    PLOT_EPOCH = 5

    dataName = data_name()
    trainDataset = train_dataset()
    valDataset = val_dataset()

    trainLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    ModelData_name = Path(__file__).name[:-3] + '_' + regmodel.model_name + '_' + dataName + '_s' + str(len(trainLoader.sampler))

    mainfile = Path(__file__).name
    dl_file = 'stdDL.py'
    # model_file1 = 'second_models.py'
    model_file2 = 'rank1Qi.py'

    parentDir = ModelData_name
    if not os.path.exists(parentDir):
        os.mkdir(parentDir)

    childDir = 'initLR' + '%.0e' % INITIAL_LEARNING_RATE + '_BS' + str(trainLoader.batch_size) + '_T' + time.strftime('%m%d_%H%M%S')
    saveFolder = os.path.join(parentDir, childDir)
    os.mkdir(saveFolder)

    copyfile(mainfile, os.path.join(saveFolder, mainfile))
    copyfile(dl_file, os.path.join(saveFolder, dl_file))
    # copyfile(model_file1, os.path.join(saveFolder, model_file1))
    copyfile(model_file2, os.path.join(saveFolder, model_file2))
    print('文件', mainfile, dl_file, model_file2, '已复制')

    cutNumber = cut_number()

    trainer = Trainer(regmodel, trainLoader, valLoader, cutNumber, INITIAL_LEARNING_RATE, STANDARD_LEARNING_RATE, TRAIN_PATIENCE, PLOT_EPOCH, saveFolder)
    trainer.train()


if __name__ == '__main__':
    global_tic = time.time()
    main()
    global_toc = time.time()
    print('该次训练用时%f' % (global_toc-global_tic))
