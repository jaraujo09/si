import sys
import numpy as np
import pandas as pd
from src.si.data.dataset import Dataset
from src.si.io.csv_file import read_csv, write_csv 
from src.si.io.data_file import *

filename = 'C:\\Users\\Fofinha\\Desktop\\UNI\\MESTRADO\\2o ANO\\Sistemas Inteligentes\\si\\datasets\\iris\\iris.csv'

#1.1.
dataset = read_csv(filename, sep = ',')

#1.2.
penultimate = dataset.X[:, -2]
penultimate.shape

#1.3.
last_10 = dataset[:-10, :]
last_10.get_mean

#1.4.
selected_samples = dataset[dataset.X <= 6]
tot = sum(selected_samples)

#1.5.
not_setosa = dataset[dataset['class'] != 'Iris-setosa']
total = sum(not_setosa)
