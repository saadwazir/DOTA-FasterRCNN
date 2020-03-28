import sys
import os
import random
from skimage import io
import pandas as pd
from matplotlib import pyplot as plt
from shutil import copyfile

import cv2
import tensorflow as tf
from pandas import read_csv
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

print(gpu_options)
print(session)


#%%

root_path = "/home/saad/DOTA_devkit/"

df = pd.DataFrame()
df = read_csv(root_path + 'train_annotation.txt',names=['image-name', 'x1', 'y1', 'x2', 'y2', 'class'],na_values=['.'])
#print(df)


#%%

img_data_single_names = pd.DataFrame()

data = df[['image-name']]
#print(data.loc[1])

temp = ""

for index, row in range( data.iterrows() ):
    img_name_1 = row['image-name']

    if temp == img_name_1:
        continue
    else:
        #print(img_name_1)
        img_data_single_names = img_data_single_names.append({'image-name':img_name_1},ignore_index=True)
        temp = img_name_1



root_path = "/home/saad/DOTA_devkit/dota-dataset-split/"

img_file = img_data_single_names.iloc[2]
img_file = img_file['image-name']

print(img_file)

