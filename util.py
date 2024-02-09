# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:46:01 2024

@author: Songho Lee
@reference: https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet
"""
#%% Utils
import os 
import numpy as np

import torch
import torch.nn as nn


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    torch.save({'net': net.state_dict(),
                'optim': optim.state_dict()},
               "%s//modelk_epoch%d.pth" % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch=0
        return net, optim, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s//%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    
    return net, optim, epoch


