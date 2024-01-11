import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from models import generate_model
from lib.spatial_transforms import *

from data.data_loader import DataSet
from lib.utils import Logger
from cls import build_model
import time
import os
import sys

from lib.utils import AverageMeter, calculate_accuracy
from torch.autograd import Variable
from torch.optim import lr_scheduler
from test_cls import test
from tst_class import tst


def get_mean(norm_value=255):
    return [114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value]


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))

    # n_correct_elems = correct.float().sum().data[0]
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def train(cur_iter, total_iter, data_loader, model, criterion, optimizer, scheduler, opt):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    i = cur_iter
    while i < total_iter:
        for _, (inputs, targets) in enumerate(data_loader):

            if not opt.no_cuda:
                targets = targets.cuda()

            targets = Variable(targets)

            inputs = Variable(inputs)
            outputs = model(inputs)

            # critreion 为交叉熵损失函数(nn.CrossEntropyLoss)
            loss = criterion(outputs, targets)

            # 待解决
            acc = calculate_accuracy(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())


            #  这里 改格式！！！！
            print('Iter:{} Loss_conf:{} acc:{} lr:{}'.format(i + 1, loss.item(), acc, optimizer.param_groups[0]['lr']),
                  flush=True)
            i += 1

            if i % 25 == 0:
                save_file_path = os.path.join(opt.result_dir, 'model_iter{}.pth'.format(i))
                print("save to {}".format(save_file_path))
                states = {
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
            if i >= total_iter:
                break

    save_file_path = os.path.join(opt.result_dir, 'model_final.pth'.format(opt.checkpoint_path))
    print("save to {}".format(save_file_path))
    states = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)


def get_lastest_model(opt):
    if opt.resume_path != '':
        return 0
    if os.path.exists(os.path.join(opt.result_dir, 'model_final.pth')):
        opt.resume_path = os.path.join(opt.result_dir, 'model_final.pth')
        return opt.total_iter

    iter_num = -1
    for filename in os.listdir(opt.result_dir):
        if filename[-3:] == 'pth':
            _iter_num = int(filename[len('model_iter'):-4])
            iter_num = max(iter_num, _iter_num)
    if iter_num > 0:
        opt.resume_path = os.path.join(opt.result_dir, 'model_iter{}.pth'.format(iter_num))
    return iter_num


# 这里写一个get_best_model(opt)


if __name__ == '__main__':
    opt = parse_opts()

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)

    opt.mean = get_mean(opt.norm_value)
    print(opt)

    torch.manual_seed(opt.manual_seed)

    model = build_model(opt, "train")

    cur_iter = 0
    if opt.auto_resume and opt.resume_path == '':
        cur_iter = get_lastest_model(opt)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        model.load_state_dict(checkpoint['state_dict'])

    parameters = model.parameters()
    criterion = nn.CrossEntropyLoss()
    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.momentum

    optimizer = optim.SGD(parameters, lr=opt.learning_rate,
                          momentum=opt.momentum, dampening=dampening,
                          weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_train:
        spatial_transform = get_train_spatial_transform(opt)
        temporal_transform = None
        target_transform = None
        training_data = DataSet(os.path.join(opt.root_dir, opt.train_subdir), opt.image_list_path,
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform,
                                target_transform=target_transform, sample_duration=opt.sample_duration)

        weights = torch.DoubleTensor(training_data.weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=opt.batch_size,
                                                           num_workers=opt.n_threads, sampler=sampler, pin_memory=True)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=60000)

        train(cur_iter, opt.total_iter, training_data_loader, model, criterion, optimizer, scheduler, opt)

    tst(opt, model)
