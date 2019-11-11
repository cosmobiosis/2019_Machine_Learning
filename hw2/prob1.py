from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import numpy as np
import os

dataset = []
with open('yeast.data') as f: #parse the data
    for line in f:
        m = line.split()
        row_vec = []
        for each in m:
            row_vec.append(each)
        dataset.append(row_vec)

for each in dataset: #cast string to data values for each sample
    for i in range(1, 9):
        each[i] = (float)(each[i])

dataset = np.matrix(dataset)
trainX = dataset[:, 1:9]

clf1 = IsolationForest()
iso_outlier = clf1.fit_predict(trainX)
print("#outliers for Iso: ", np.count_nonzero(iso_outlier == -1))

clf2 = LocalOutlierFactor(n_neighbors  = 8)
lof_outlier = clf2.fit_predict(trainX)
print("#outliers for LOF: ", np.count_nonzero(lof_outlier == -1))

outlier_difference = abs(iso_outlier - lof_outlier)
print("number of disagreement is: ", np.count_nonzero(outlier_difference != 0))

new_dataset = []
for i in range(len(dataset)):
    if iso_outlier[i] == 1:
        new_dataset.append(dataset[i])