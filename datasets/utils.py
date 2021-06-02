#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2021 - zihao.chen
'''
@Author : zihao.chen
@File : utils.py 
@Create Date : 2021/6/2
@Descirption :
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
from tqdm import tqdm
import numpy as np
import cv2
import os
import random
from selene.model.crnn.utils.ctcdecode import be_decode
from selene.model.crnn.utils.tools_wordmap import get_map
import selene.model.crnn.reg_config as params
from torchvision.transforms import transforms


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
        self.alphabet.append('-')  # for `-1` index
        self.map_dict = get_map()

    def encode(self, text, map_flag=False):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0]) == bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                if char in self.dict:
                    index = self.dict[char]
                    result.append(index)
                else:
                    if map_flag:
                        if char in self.map_dict:
                            char = self.map_dict[char]
                            if char in self.dict:
                                index = self.dict[char]
                                result.append(index)
                            else:
                                result.append(self.dict[' '])
                        else:
                            result.append(self.dict[' '])
                    else:
                        result.append(self.dict[' '])
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    def beam_search_decode(self, prediction):
        labels, scores = be_decode(prediction)
        print(labels)
        text = ""
        for la in labels:
            text += self.alphabet[la - 1]
        return text


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)
    # print(v.size())


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


def to_alphabet(path, gate=20):
    result = {}
    for dir_ in path:
        with open(dir_, 'r', encoding='utf-8') as file:
            for word in file.readlines():
                word = word.strip().split('\t')
                if len(word) == 2:
                    for alp in word[1]:
                        if alp in result:
                            result[alp] += 1
                        else:
                            result[alp] = 0
    alphabet = []
    for k, v in result.items():
        if v > gate:
            alphabet.append(k)
    with open('char.txt', 'w', encoding='utf-8') as fw:
        for apl in alphabet:
            fw.write(apl)
            fw.write("\n")
    return alphabet


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label


def compute_std_mean(txt_path, image_prefix, NUM=None):
    imgs = np.zeros([params.imgW, params.imgW, 1, 1], dtype=np.uint8)
    means, stds = [], []
    with open(txt_path, 'r') as file:
        contents = [c.strip().split('\t')[0] for c in file.readlines()]
        if NUM is None:
            NUM = len(contents)
        else:
            random.shuffle(contents)
        for i in tqdm(range(NUM)):
            file_name = contents[i]
            img_path = os.path.join(image_prefix, file_name)

            if not os.path.exists(img_path):
                print(img_path)
                continue
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            img = cv2.resize(img, (0, 0), fx=params.imgW / w, fy=params.imgW / h, interpolation=cv2.INTER_CUBIC)
            img = img[:, :, np.newaxis, np.newaxis]
            # print(img.sum())
            imgs = np.concatenate((imgs, img), axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(1):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()
    # print(means, stds)

    return stds, means


def compute_buckets_std_mean(txt_paths, image_prefixs, NUM=None):
    means, stds = [], []
    pixels = np.array([], dtype=np.uint8)
    img_buckets = []
    mask_buckets = []
    for txt_path, image_prefix in zip(txt_paths, image_prefixs):
        imgH = 32
        imgW = int(image_prefix[-2:]) * imgH

        with open(txt_path, 'r') as file:
            contents = [c.strip().split('\t')[0] for c in file.readlines()]
            if NUM is None:
                NUM = len(contents)
            else:
                random.shuffle(contents)
            imgs = np.zeros([imgH, imgW, 1, NUM], dtype=np.uint8)
            masks = np.zeros([imgH, imgW, 1, NUM], dtype=np.uint8)
            for i in tqdm(range(NUM)):
                try:
                    file_name = contents[i]
                    img_path = os.path.join(image_prefix, file_name)
                    # print(img_path)
                    if not os.path.exists(img_path):
                        print(img_path)
                        continue
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ih, iw = img.shape[:2]
                    nw = int(imgH * iw / ih)
                    img = cv2.resize(img, (nw, imgH), interpolation=cv2.INTER_CUBIC)
                    if nw > imgW:
                        img = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
                        masks[:, :, 0, i] = 1
                    else:
                        masks[:, :nw, 0, i] = 1
                        img = cv2.copyMakeBorder(img, 0, 0, 0, imgW - nw, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    # img = img[:, :, np.newaxis, np.newaxis]
                    imgs[:, :, 0, i] = img[:]

                    # imgs = np.concatenate((imgs, img), axis=3)
                except Exception as e:
                    print(img.shape)
                    print(imgs.shape)

        # imgs = imgs.astype(np.float32) / 255.
        _imgs = imgs.astype(np.float32) / 255.
        print(np.mean(_imgs[masks == 1]))
        print(np.std(_imgs[masks == 1]))
        img_buckets.append(imgs)
        mask_buckets.append(masks)
        # pixels = np.concatenate((pixels, imgs.flatten()))
    pixels = np.concatenate(img_buckets, axis=1)
    pixel_masks = np.concatenate(mask_buckets, axis=1)
    for i in range(1):
        # _pixels = pixels.ravel()
        # _mask = pixel_masks.ravel()
        _pixels = pixels.astype(np.float32) / 255.
        means.append(np.mean(_pixels[pixel_masks == 1]))
        stds.append(np.std(_pixels[pixel_masks == 1]))

    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()
    # print(means, stds)

    return stds, means


def compute_buckets_std_mean_append(txt_paths, image_prefixs, NUM=None):
    means, stds = [], []
    pixels = np.array([], dtype=np.uint8)
    img_buckets = []
    for txt_path, image_prefix in zip(txt_paths, image_prefixs):
        imgH = 32
        imgW = int(image_prefix[-2:]) * imgH

        with open(txt_path, 'r') as file:
            contents = [c.strip().split('\t')[0] for c in file.readlines()]
            if NUM is None:
                NUM = len(contents)
            else:
                random.shuffle(contents)
            imgs = np.zeros([imgH, imgW, 1, NUM], dtype=np.uint8)
            for i in tqdm(range(NUM)):
                try:
                    file_name = contents[i]
                    img_path = os.path.join(image_prefix, file_name)
                    # print(img_path)
                    if not os.path.exists(img_path):
                        print(img_path)
                        continue
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ih, iw = img.shape[:2]
                    nw = int(imgH * iw / ih)
                    img = cv2.resize(img, (nw, imgH), interpolation=cv2.INTER_CUBIC)
                    if nw > imgW:
                        img = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
                    else:
                        img = cv2.copyMakeBorder(img, 0, 0, 0, imgW - nw, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    # img = img[:, :, np.newaxis, np.newaxis]
                    _img = img.astype(np.float32) / 255.
                    means.append(np.mean(_img.ravel()))
                    stds.append(np.std(_img.ravel()))
                    # imgs[:, :, 0, i] = img[:]

                    # imgs = np.concatenate((imgs, img), axis=3)
                except Exception as e:
                    print(img.shape)
                    print(imgs.shape)

        # imgs = imgs.astype(np.float32) / 255.
        # img_buckets.append(imgs)
        # pixels = np.concatenate((pixels, imgs.flatten()))
    # pixels = np.concatenate(img_buckets,axis=1)
    # for i in range(1):
    #     _pixels = pixels.ravel()
    #     _pixels = pixels.astype(np.float32) / 255.
    #     means.append(np.mean(_pixels))
    #     stds.append(np.std(_pixels))
    stds = np.mean(stds)
    means = np.mean(means)
    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()
    # print(means, stds)

    return stds, means


def read_alphabet(path):
    alphabet = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            alphabet.append(line.strip("\n"))
    return alphabet


def toTensorImage(image, is_cuda=True):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).unsqueeze(0)
    if is_cuda:
        image = image.cuda()
    return image


def toTensor(item, is_cuda=True):
    item = torch.Tensor(item)
    if is_cuda:
        item = item.cuda()
    return item


def rotate(img, angle):
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation


def detection_resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (img_size[0] / new_h, img_size[1] / new_w)


def mk_not_exits_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    random.seed(24)
    np.random.seed(24)
    img_roots = ["/data2/zihao.chen/data/train_data/recognition/picture_10"]
    label_paths = ["/data2/zihao.chen/data/train_data/recognition/picture_10.txt"]
    stds, means = compute_std_mean(label_paths[0], img_roots[0], 1000)
    print(stds, means)
