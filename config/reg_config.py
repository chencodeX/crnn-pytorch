#!/usr/bin/python3
# -*- coding:utf-8 _*-
# Copyright (c) 2021 - zihao.chen
'''
@Author : zihao.chen
@File : reg_config.py
@Create Date : 2021/6/2
@Descirption :
'''

manualSeed = 24
log_dir = "./logging"
char_dir = "./datas/crnn/chars.txt"
# resume_dir = './base_info_expr/crnn_warmup_model_new_2_0.62145625.pth'
resume_dir = ''
model_flag = 'crnn_dynamic_model_0601'
experiment = './base_info_expr'
data_base_path = "/datafaster/zihao.chen/data/train_data/recognition"

alphabet = None
random_sample = True
best_val_accuracy = 0.0
best_test_accuracy = 0.0
keep_ratio = False
adam = True
adadelta = False
saveInterval = 1000
valInterval = 400
n_test_disp = 10
displayInterval = 1000
val_number = 10000

# model parameter
beta1 = 0.5
lr = 0.001
warmup = False
warmup_lr = 0.00003
warmup_step = 50000
niter = 300
nh = 256
imgW = 640
imgH = 32
val_batchSize = 64
batchSize = 52
workers = 8
# std = 0.193
# mean = 0.588
std = 0.301
mean = 0.808
dynamic = True