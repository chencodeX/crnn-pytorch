#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2020 - zihao.chen
'''
@Author : zihao.chen
@File : reg_dataloader.py
@Create Date : 2020/11/2
@Descirption :
'''
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import os
import random
import time
import re
import math

import config.reg_config as reg_config
from datasets.tools.data_augument import pepper_and_salt, sharpening, gaussian_blur, normalize, random_erode, \
    scan, random_invert


class OCRDataset(Dataset):
    def __init__(self, img_root, label_path, resize, val=False, dynamic=False):
        super(OCRDataset, self).__init__()
        self.labels = self.get_labels(label_path, img_root)
        self.height, self.width = resize
        if dynamic:
            self.mode = 3
        else:
            self.mode = 2
        self.aug = True

    @staticmethod
    def get_labels(label_path, image_path):
        labels = []
        label_names = os.listdir(label_path)
        label_paths = [os.path.join(label_path, file_name) for file_name in label_names]
        image_paths = [lp.replace('txt', 'jpg') for lp in label_paths]
        for la_path, im_path in zip(label_paths, image_paths):
            with open(la_path, 'rb') as file:
                temp_lins = file.readlines()
                for line in temp_lins:
                    word = line.decode('gb18030').strip('\n')
                    labels.append({im_path: word})
        return labels

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def pre_processing(img):
        # already have been computed
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img.sub_(reg_config.mean).div_(reg_config.std)
        return img

    def ocr_preprocess(self, img, w, h):
        ih, iw = img.shape[0:2]
        nw = int(h * iw / ih)
        if nw < 1:
            nw = 1
        img = cv2.resize(img, (nw, h), interpolation=cv2.INTER_CUBIC)
        if nw > w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.copyMakeBorder(img, 0, 0, 0, w - nw, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = (np.reshape(img, (32, w, 1))).transpose(2, 0, 1)
        img = self.pre_processing(img)
        return img

    def ocr_dynamic_preprocess(self, img, w, h):
        ih, iw = img.shape[0:2]
        nw = int(h * iw / ih)
        if nw < 1:
            nw = 1
        img = cv2.resize(img, (nw, h), interpolation=cv2.INTER_CUBIC)
        if nw > w:
            nw = w
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            if random.random() < 0.5:
                img = cv2.copyMakeBorder(img, 0, 0, 0, w - nw, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            else:
                img = cv2.copyMakeBorder(img, 0, 0, 0, w - nw, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = (np.reshape(img, (32, w, 1))).transpose(2, 0, 1)
        seq_len = math.ceil(nw / 4)
        img = self.pre_processing(img)
        return img, seq_len

    def __getitem__(self, ind):
        while True:
            try:

                image_name = list(self.labels[ind].keys())[0]
                image = cv2.imread(image_name)
                h, w, c = image.shape
                # if h > w:
                image = image.transpose((1, 0, 2))
                if image is None:
                    print(self.labels[ind])
                    ind += 1
                    ind = ind % len(self.labels)
                    continue
                if (image.max() - image.min()) < 64:
                    image = normalize(image)
                if (not self.val) and random.random() <= 0.7:
                    aug_ind = random.randint(0, 6)
                    if aug_ind == 0:
                        sigma = random.uniform(0.01, 0.04)
                        image = pepper_and_salt(image, sigma)
                    elif aug_ind == 1:
                        image = sharpening(image)
                    elif aug_ind == 2:
                        sigma = random.uniform(0.5, 1.2)
                        image = gaussian_blur(image, sigma)
                    elif aug_ind == 3:
                        image = random_erode(image)
                    elif aug_ind == 4:
                        image = scan(image)
                    # elif aug_ind == 5:
                    #     image = random_crop(image)
                    else:
                        # image = laplacian_sharpen(image)
                        image = random_invert(image)

                if self.mode == 1:
                    h, w = image.shape
                    image = cv2.resize(image, (0, 0), fx=self.width / w, fy=self.height / h,
                                       interpolation=cv2.INTER_CUBIC)
                    image = (np.reshape(image, (32, self.width, 1))).transpose(2, 0, 1)
                    image = self.pre_processing(image)
                elif self.mode == 2:
                    image = self.ocr_preprocess(image, self.width, self.height)
                elif self.mode == 3:
                    image, seq_len = self.ocr_dynamic_preprocess(image, self.width, self.height)
                    return image, seq_len, ind
            except Exception as e:
                print('image error :', self.labels[ind])
                print(e)
                ind += 1
                ind = ind % len(self.labels)
                continue
            return image, ind


if __name__ == '__main__':
    # random seed
    random.seed(reg_config.manualSeed)
    np.random.seed(reg_config.manualSeed)
    torch.manual_seed(reg_config.manualSeed)

    img_roots = "/datafaster/zihao.chen/data/train_data/recognition/imgs"
    label_paths = "/datafaster/zihao.chen/data/train_data/recognition/labels"
    im_dir = []
    label_dir = []
    dataset = OCRDataset(img_roots, label_paths, (reg_config.imgH, reg_config.imgW), val=False,
                         dynamic=True)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    start = time.time()
    for data_20 in data_loader:
        print(data_20[0].size())
        print(data_20[1].size())
        seq_lengths = torch.LongTensor(data_20[1].long())
        print(type(data_20[1]))
        print(data_20[2].size())
        print(type(data_20[2]))
