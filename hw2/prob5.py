from utility_funcs import *
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import IsolationForest

if not os.path.isdir("training"):
    os.mkdir('training')
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

dataset = np.array(dataset)
trainX = dataset[:, 1:9]
clf = IsolationForest()
iso_outlier = clf.fit_predict(trainX)

new_dataset = []
for i in range(len(dataset)):
    if iso_outlier[i] == 1:
        new_dataset.append(dataset[i])

dataset = np.array(new_dataset)
trainX = dataset[0: (int)(0.66 * len(dataset)), 1:9]
trainY = dataset[0: (int)(0.66 * len(dataset)),9]
trainY = encoding(trainY)

testX = dataset[(int)(0.66 * len(dataset)):, 1:9]
testY = dataset[(int)(0.66 * len(dataset)):, 9]
testY = encoding(testY)

num_epochs = 500

for num_hidden_layers in range(1, 4):
    for num_hidden_nodes in ([3, 6, 9, 12]):
        model = vary_train_ANN(trainX, trainY, num_epochs, num_hidden_layers, num_hidden_nodes)
        error = get_misclassified_ratio(model, testX, testY)
        print("Layer ", num_hidden_layers, " Node ", num_hidden_nodes, "is \n", error)
        #print("Layer ", num_hidden_layers, " Node ", num_hidden_nodes, "is \n",
              #model.evaluate(np.asarray(testX), np.asarray(testY), verbose=1))