import os
import pandas as pd
import csv
from pandas import read_csv
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

entries = os.listdir('dota-dataset-split/labels-un')
print(entries)

root_path = "dota-dataset-split/labels-un/"


df1 = pd.DataFrame()


for i in entries:
    df = read_csv(root_path + i, names=['image-name', 'x1', 'y1', 'x2', 'y2', 'class'],na_values=['.'])
    df1 = df1.append(df,ignore_index=True)

print(df1)

export_csv = df1.to_csv(r'annotation-new.txt', index = None, header=False)

#base_path = "/content/drive/My Drive/dota-dataset/"

df = pd.read_csv("annotation-new.txt", names=['image-name', 'x1', 'y1', 'x2', 'y2', 'class'])
print(df)

first_split = df.sample(frac=0.7)
print(first_split)

second_split=df.drop(first_split.index)
print(second_split)

#third_split = second_split.sample(frac=0.02)
#print(third_split)


export_csv = first_split.to_csv(r'train_annotation.txt', index = None, header=False)
export_csv = second_split.to_csv(r'test_annotation.txt', index = None, header=False)


#base_path = "/content/drive/My Drive/dota-dataset/"

df = pd.read_csv("train_annotation.txt", names=['image-name', 'x1', 'y1', 'x2', 'y2', 'class'])
print(df)

df = pd.read_csv("test_annotation.txt", names=['image-name', 'x1', 'y1', 'x2', 'y2', 'class'])
print(df)