import sys
import numpy as np
import pandas as pd
from si.data.dataset import Dataset
from si.io.csv_file import read_csv, write_csv 
from si.io.data_file import *

filename = 'C:\\Users\\Fofinha\\Desktop\\UNI\\MESTRADO\\2o ANO\\Sistemas Inteligentes\\si\\datasets\\iris\\iris.csv'

#1.1.
dataset = read_csv(filename, sep = ',', features = True, label = True)
print(dataset.shape())

#1.2.
penultimate = dataset.X[:, -2]
print(penultimate.shape)

#1.3.
last_10 = dataset.X[-10:]
print(last_10.mean())


#1.4.
selected_samples = len(dataset.X[dataset.X <= 6])
print(selected_samples)

#1.5.
not_setosa = len(dataset.X[dataset.y!= 'Iris-setosa'])
print(not_setosa)

#2.
