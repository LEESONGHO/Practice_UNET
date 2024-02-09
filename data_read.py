# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:24:23 2024

@author: Songho Lee
@reference: https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet
"""

#%% Library
import os 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
set_seeds(42)

#%% Path
path_cwd = os.getcwd()
print("path_cwd: ", path_cwd)

#%% Dataset
dir_data = path_cwd+'\\datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

print("ny = ",ny,"\t| nx = ",nx)
print("nframe = ", nframe)

#%% Train - val - test
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
    
dir_save_val = os.path.join(dir_data, 'val')
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
    
dir_save_test = os.path.join(dir_data, 'test')
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# All
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# train
offset_nframe = 0
for i in range(nframe_train):
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])
    
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)
    
# val
offset_nframe = nframe_train
for i in range(nframe_val):
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])
    
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# test
offset_nframe = nframe_train + nframe_val
for i in range(nframe_test):
    img_label.seek(id_frame[i+offset_nframe])
    img_input.seek(id_frame[i+offset_nframe])
    
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)


#%% Check
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()

#%%
print("DONE")





