#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2021 - zihao.chen <chenzihao@mokahr.com> 
'''
@Author : zihao.chen
@File : trainer.py 
@Create Date : 2021/6/3
@Descirption :
'''
from __future__ import print_function
from torch.utils.data import DataLoader
import random
import torch
import torch.nn as nn
from torch.nn import init
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import numpy as np
import os
import time

import datasets.utils as utils
import models.crnn_mv3 as crnn
import config.reg_config as reg_config
from datasets.reg_dataloader import OCRDataset
from tensorboardX import SummaryWriter


def compute_acc_rough(label, prediction):
    label_t = label.replace("。", ".").replace("，", ",").replace("；", ";").replace("：", ":") \
        .replace("（", "(").replace("）", ")").replace(" ", "").replace("*", "").replace("#", "").replace("、", ",")
    prediction_t = prediction.replace("。", ".").replace("，", ",").replace("；", ";"). \
        replace("：", ":").replace("（", "(").replace("）", ")").replace(" ", "").replace("*", "").replace("#",
                                                                                                        "").replace("、",
                                                                                                                    ",")
    return float(label_t == prediction_t)


def val(crnn, val_loader, data_set, criterion, iteration):
    print('Start val...')
    for p in crnn.parameters():
        p.requires_grad = False
    crnn.eval()
    with torch.no_grad():
        n_correct = 0
        n_rough_correct = 0
        loss_avg = utils.averager()
        i_batch = 0
        # preds_size = None
        for data_40 in val_loader:
            if reg_config.dynamic:
                image = data_40[0].to(device)
                seq_lens = data_40[1]
                label = utils.get_batch_label(data_set, data_40[2])
                preds = crnn(image, seq_lens)
            else:
                image = data_40[0].to(device)
                label = utils.get_batch_label(data_set, data_40[1])
                preds = crnn(image)

            batch_size = image.size(0)
            text, length = converter.encode(label, map_flag=True)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            cost = criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, label):
                if pred == target:
                    n_correct += 1
                n_rough_correct += compute_acc_rough(pred, target)

            if (i_batch + 1) % reg_config.displayInterval == 0:
                print('[%d/%d][%d/%d]' % (iteration, reg_config.niter, i_batch, len(val_loader)))
            if i_batch == reg_config.val_number:
                break
            i_batch += 1

        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:reg_config.n_test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
            print('%-10s => %-10s, gt: %-10s' % (raw_pred, pred, gt))

        print(n_correct)
        sum_number = reg_config.val_batchSize * reg_config.val_number
        accuracy = n_correct / sum_number
        rough_acc = n_rough_correct / sum_number
        print('Test loss: %f, Acc: %f rough acc: %f' % (loss_avg.val(), accuracy, rough_acc))
        writer.add_scalar('test Loss', loss_avg.val(), iteration)
        writer.add_scalar('test acc', rough_acc, iteration)
    return accuracy, rough_acc, loss_avg.val()


def train(crnn, train_loader, criterion, iteration, optimizer, warm_up=False):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()
    loss_avg = utils.averager()
    i_batch = 0
    for data_40 in train_loader:
        if warm_up and i_batch % 100 == 0:
            adjust_learning_rate(optimizer, 0, warm_up, batch_idx=i_batch)
        if reg_config.dynamic:
            image = data_40[0].to(device)
            seq_lens = data_40[1]
            label = utils.get_batch_label(dataset_bucket_train, data_40[2])
            preds = crnn(image, seq_lens)
        else:
            image = data_40[0].to(device)
            label = utils.get_batch_label(dataset_bucket_train, data_40[1])
            preds = crnn(image)
        batch_size = image.size(0)
        text, length = converter.encode(label, map_flag=True)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)
        if (i_batch + 1) % reg_config.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (iteration, reg_config.niter, i_batch, len(train_loader), loss_avg.val()))
            writer.add_scalar('Train Loss', loss_avg.val(), iteration * len(train_loader) + i_batch)
            loss_avg.reset()

        if (i_batch + 2) % 1000 == 0:
            torch.save(crnn.state_dict(), '{0}/{1}_{2}_{3}_{4}.pth'.
                       format(reg_config.experiment, reg_config.model_flag, iteration, i_batch, loss_avg.val()))
        i_batch += 1
    print('[%d/%d][%d/%d] Loss: %f' %
          (iteration, reg_config.niter, i_batch, len(train_loader), loss_avg.val()))
    writer.add_scalar('Train Loss', loss_avg.val(), iteration * len(train_loader) + i_batch)
    loss_avg.reset()


def adjust_learning_rate(optimizer, epoch, warmup=False, batch_idx=10000):
    """
    Parameters
    ----------
    optimizer 优化器
    epoch  迭代次数
    warmup 是否使用 warmup 策略
    batch_idx warmup需要的当前步数

    Returns
    -------

    """
    if warmup:
        if batch_idx <= reg_config.warmup_step:
            warmup_percent_done = batch_idx / reg_config.warmup_step
            lr_ = max(reg_config.lr * warmup_percent_done, reg_config.warmup_lr)
        else:
            return
    else:
        lr_ = reg_config.lr * (0.5 ** (epoch // 10))
        lr_ = max(lr_, 0.00001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


def main(crnn, train_loader, val_loader, criterion, optimizer):
    crnn = crnn.to(device)
    criterion = criterion.to(device)
    iteration = 0
    while iteration < reg_config.niter:
        start_time = time.time()
        print("Epoch {0} time log: {1}".format(iteration, time.ctime()))
        adjust_learning_rate(optimizer, epoch=iteration)
        warmup = False
        if reg_config.warmup and iteration == 0:
            warmup = True
        train(crnn, train_loader, criterion, iteration, optimizer, warmup)
        end_time = time.time()
        print("Epoch {0} time log: {1}".format(iteration, time.ctime()))
        print("Epoch: {0} spend time: {1}".format(iteration, end_time - start_time))

        accuracy, rough_acc, loss_ = val(crnn, val_loader, dataset_bucket_val, criterion, iteration)

        if accuracy > reg_config.best_val_accuracy:
            torch.save(crnn.state_dict(),
                       '{0}/{1}_new_{2}_{3}.pth'.format(reg_config.experiment, reg_config.model_flag, iteration,
                                                        accuracy))
            torch.save(crnn.state_dict(),
                       '{0}/{1}_best_process.pth'.format(reg_config.experiment, reg_config.model_flag))

        print("is best accuracy: {0}".format(accuracy > reg_config.best_val_accuracy))
        iteration += 1


def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0  # replace all nan/inf in gradients to zero


if __name__ == '__main__':
    # random seed
    random.seed(reg_config.manualSeed)
    np.random.seed(reg_config.manualSeed)
    torch.manual_seed(reg_config.manualSeed)
    cudnn.benchmark = True

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # logger
    writer = SummaryWriter(reg_config.log_dir)

    # dict
    alphabet = utils.read_alphabet(reg_config.char_dir)
    print("Word Vocab: {}".format(len(alphabet)))
    nclass = len(alphabet) + 1

    # model load
    # crnn = crnn.CRNN(32, 1, nclass, reg_config.nh)
    crnn = crnn.CRNN(reg_config.nh, nclass, dynamic=reg_config.dynamic)
    # crnn.apply(weights_init)

    if reg_config.resume_dir != '':
        print('loading pretrained model from %s' % reg_config.resume_dir)
        pretrained_dict = torch.load(reg_config.resume_dir)
        # 注释部分为部分权重加载时用
        # pretrained_dict.pop("rnn.1.embedding.1.weight")
        # pretrained_dict.pop("rnn.1.embedding.1.bias")
        # model_dict = crnn.state_dict()
        # model_dict.update(pretrained_dict)
        crnn.lstm_r.lstm_2.embedding = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(reg_config.nh * 2, 6894))
        crnn.load_state_dict(pretrained_dict, strict=False)
        crnn.lstm_r.lstm_2.embedding = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(reg_config.nh * 2, 1911))

        init.normal_(crnn.lstm_r.lstm_2.embedding[1].weight, std=0.001)
        init.constant_(crnn.lstm_r.lstm_2.embedding[1].bias, 0)

    # data loader process
    img_roots = "/datafaster/zihao.chen/data/train_data/datasets_for_CRNN/new_set"
    label_paths = "/datafaster/zihao.chen/data/train_data/datasets_for_CRNN/label_txt.txt"

    # store model path
    utils.mk_not_exits_dir(reg_config.experiment)

    # read train set
    dataset_bucket_train = OCRDataset(img_roots, label_paths, (reg_config.imgH, reg_config.imgW),
                                      dynamic=True)
    dataset_bucket_val = OCRDataset(img_roots, label_paths, (reg_config.imgH, reg_config.imgW), val=True, dynamic=True)

    loader_bucket_train = DataLoader(dataset_bucket_train, batch_size=reg_config.batchSize, shuffle=False,
                                     num_workers=reg_config.workers)
    loader_bucket_val = DataLoader(dataset_bucket_val, batch_size=reg_config.val_batchSize, shuffle=False,
                                   num_workers=reg_config.workers)

    converter = utils.strLabelConverter(alphabet)
    print("network class: {0}".format(nclass))
    criterion = torch.nn.CTCLoss(reduction='sum')

    # setup optimizer
    if reg_config.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=reg_config.lr,
                               betas=(reg_config.beta1, 0.999))
    elif reg_config.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=reg_config.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=reg_config.lr)

    crnn.register_backward_hook(backward_hook)
    main(crnn, loader_bucket_train, loader_bucket_val, criterion, optimizer)
