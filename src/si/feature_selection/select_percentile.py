import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:
    def __init__(self, score_func = f_classification, percentile, F, p):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset : Dataset):
        self.F, self.p = self.score_func(dataset)
        return self
    
    def transform(self):
        