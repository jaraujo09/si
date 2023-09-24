from dataset import Dataset
import numpy as np

X = np.array([[1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0]])

y = np.array([0, 1, 2])

dataset = Dataset(X, y)
print(dataset.dropna().shape())

X = np.array([[1.0, 2.0, 3.0],
                [4.0, np.nan, 6.0],
                [7.0, 8.0, 9.0]])

y = np.array([0, 1, 2])

dataset = Dataset(X, y)
print(dataset.fillna([4.0, 5.0, 6.0]).X)

X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([1, 2])
features = np.array(['a', 'b', 'c'])
label = 'y'
dataset = Dataset(X, y, features, label)
print(dataset.remove_from_index(1).X)