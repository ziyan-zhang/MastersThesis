"""-----------------------------------------------------
创建时间 :  2020/05/25  21:02
todo    :
-----------------------------------------------------"""
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as tvtransforms
from progressbar import ProgressBar

def get_transforms_train():
    transforms_list = [
        # tvtransforms.RandomHorizontalFlip(),
        # tvtransforms.RandomVerticalFlip(),
        tvtransforms.ToTensor(),
    ]
    transformscp = tvtransforms.Compose(transforms_list)
    return transformscp

def get_transforms_val():
    transforms_list = [
        tvtransforms.ToTensor(),
    ]
    transformscp = tvtransforms.Compose(transforms_list)
    return transformscp


def get_dist_type(img_path):
    distortion_str = img_path.split('\\')[-1][3:6]
    if distortion_str in ['002', '003', '004']:
        return 0
    elif distortion_str in ['005', '006', '007']:
        return 1
    elif distortion_str in ['008', '009']:
        return 2
    elif distortion_str in ['010', '011']:
        return 3
    elif distortion_str in ['012', '013', '001']:
        return 4
    else:
        raise ValueError('图像名称不对，导致失真类型错误!')


def get_dist_rank(img_path):
    distortion_str = img_path.split('\\')[-1][3:6]
    if distortion_str in ['002', '005', '009', '011', '001']:
        return 0
    elif distortion_str in ['003', '006', '008', '010', '012']:
        return 1
    elif distortion_str in ['004', '007', '013']:
        return 2
    else:
        raise ValueError('图像名称不对，导致失真等级错误！')


class MyDataset(Dataset):
    def __init__(self, img_folder, labelFpath, transforms):
        super(MyDataset, self).__init__()
        self.img_pathes = [os.path.join(img_folder, fileName) for fileName in os.listdir(img_folder)]
        self.img_pathes.sort()
        self.transforms = transforms

        label_pd = pd.read_csv(labelFpath, header=None)
        label_array = np.array(label_pd)
        self.label_array = label_array.squeeze()

        self.img_reads = []
        self.distType = []
        self.distRank = []
        print('加载到内存。。。')
        progress = ProgressBar()
        for i in progress(range(len(self.img_pathes))):
            img_path_i = self.img_pathes[i]
            img_read_i = Image.open(img_path_i)
            dist_type_index = get_dist_type(img_path_i)
            dist_rank_index = get_dist_rank(img_path_i)
            self.distType.append(dist_type_index)
            self.distRank.append(dist_rank_index)

            if self.transforms:
                img_read_i = self.transforms(img_read_i)
            self.img_reads.append(img_read_i)

        assert len(self.img_reads) == len(self.label_array)
        assert len(self.img_reads) == len(self.distType)

    def __getitem__(self, index):
        return self.img_reads[index], self.distType[index], self.distRank[index], self.label_array[index]

    def __len__(self):
        return len(self.img_reads)

_trainImageFolder = 'D:\\ziyan\\July2_cut_Overlap\\train_img'
_trainLabelPath = 'D:\\ziyan\\July2_cut_Overlap\\mos_train7_16.csv'
_valImageFolder = 'D:\\ziyan\\July2_cut_Overlap\\val_img'
_valLabelPath = 'D:\\ziyan\\July2_cut_Overlap\\mos_val3_16.csv'


def train_dataset():
    return MyDataset(_trainImageFolder, _trainLabelPath, get_transforms_train())


def val_dataset():
    return MyDataset(_valImageFolder, _valLabelPath, get_transforms_val())


def cut_number():
    return 16


def data_name():
    return Path(__file__).name[:-3]


if __name__ == '__main__':
    ii = train_dataset()[0]
