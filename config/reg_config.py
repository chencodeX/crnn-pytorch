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
char_dir = "./datasets/chars_25.txt"
# resume_dir = './base_info_expr/crnn_dynamic_model_0208_new_6_0.93200625.pth'
resume_dir = ''
model_flag = 'crnn_dynamic_model_0621'
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
displayInterval = 10000
val_number = 40

# model parameter
beta1 = 0.5
lr = 0.01
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
std = [0.16319881, 0.14354597, 0.13519511]
mean = [0.46323372, 0.54948276, 0.61660614]
dynamic = True
