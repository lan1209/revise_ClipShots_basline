import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import generate_model
import os


def build_model(opt, phase):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    
    num_classes = opt.n_classes
    model = generate_model(opt)

    if phase == 'train' and opt.pretrain_path:
        model.load_weights(opt.pretrain_path)

    model = model.cuda()
    if phase == 'train':
        model = nn.DataParallel(model, device_ids=range(opt.gpu_num))
    else:
        model = nn.DataParallel(model, device_ids=range(1))
    
    return model
