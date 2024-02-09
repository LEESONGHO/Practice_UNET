# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:17:10 2024

@reference: https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet
"""
#%% Library
import os
import numpy as np
import matplotlib.pyplot as plt

#%% Path
path_cwd = os.getcwd()
print("path_cwd: ", path_cwd)

#%% load
result_dir = path_cwd + '\\result\\numpy' # './results/numpy'

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

#%%
print("len(lst_label): ", len(lst_label))

#%% 
id = 2

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

#%% figure
plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(132)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()

