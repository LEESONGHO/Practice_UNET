# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:44:59 2024

@author: Songho Lee
@reference: https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet
"""

#%% Library
import argparse

import os 
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

# we made these
from model import UNet
from dataset import *
from util import *

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
set_seeds(42)

#%% Path
path_cwd = os.getcwd()
print("path_cwd: ", path_cwd)


#%% Parser - 
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default=path_cwd+'\\datasets', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default=path_cwd+'\\checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default=path_cwd+'\\log', type=str, dest="log_dir")
parser.add_argument("--result_dir", default=path_cwd+'\\result', type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

#%% Hyperparameter
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
# mode = 'test'
train_continue = args.train_continue

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

#%% device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # "0,1" Set the GPUs 0 and 1 to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

#%% Directory
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))
    
#%% Train & Eval & Test dataset

if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5),
                                   RandomFlip(),
                                   ToTensor()])
    
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0) # 8
    
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Etc
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)
    
    num_batch_train = np.ceil(num_data_train/batch_size)
    num_batch_val = np.ceil(num_data_val/batch_size)
    
else: 
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5),
                                    ToTensor()])
    
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Etc
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test/batch_size)
    
    
#%% Networks
net = UNet().to(device)

# loss
fn_loss = nn.BCEWithLogitsLoss().to(device)

# optimizer
optim = torch.optim.Adam(net.parameters(), lr=lr)

# Etc fns
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std: (x*std) + mean
fn_class = lambda x: 1.0 * (x>0.5)


#%% Tensorboard Summarywriter
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))


#%% Start Training!! or Testing!!
st_epoch = 0

if mode == 'train': # Training
    # checkpoint
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    
    # Epoch
    for epoch in range(st_epoch+1, num_epoch+1):
        net.train()
        loss_arr = []
        
        # Batch
        for batch, data in enumerate(loader_train, 1):
            # forward
            label = data['label'].to(device)
            input = data['input'].to(device)
            
            output = net(input)
            
            # loss & update
            optim.zero_grad()
            
            loss = fn_loss(output, label)
            loss.backward()
            
            optim.step()

            loss_arr += [loss.item()]
            
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))
            
            # tensorboard
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))
            
            writer_train.add_image('label', label, num_batch_train * (epoch-1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
        
        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # loss
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
    
else: # Testing
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    
    with torch.no_grad():
        net.eval()
        loss_arr = []
        
        for batch, data in enumerate(loader_test, 1):
            # forward
            label = data['label'].to(device)
            input = data['input'].to(device)
            
            output = net(input)
            
            # loss
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]
            
            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))
            
            # Tensorboard 
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))
            
            # save
            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))
        
#%%
print("DONE")
