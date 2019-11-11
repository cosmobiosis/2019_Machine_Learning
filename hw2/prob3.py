from utility_funcs import *
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
trainX = dataset[:, 1:9]
trainY = dataset[:, 9]
trainY = encoding(trainY)

num_epochs = 250
model = train_ANN(trainX, trainY, num_epochs)
err = get_misclassified_ratio(model, trainX, trainY)
print('Training Error for final model is: ', err)

last_layer_weights = model.layers[2].get_weights()[0]
last_layer_bias = model.layers[2].get_weights()[1][0]

CYT_weight_values = []
for input_index in range(3):
    kth_node_weight = last_layer_weights[input_index][0]
    CYT_weight_values.append(kth_node_weight)

print("Weight is: ")
print(CYT_weight_values)
print("Bias is: ")
print(last_layer_bias)