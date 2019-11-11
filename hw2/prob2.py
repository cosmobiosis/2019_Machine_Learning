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

num_epochs = 1000
model = train_ANN(trainX, trainY, num_epochs)

CYT_weight_all_iterations = []

#Concat all iterations of input weights for 1st Node CYT
for i in range(1, num_epochs + 1):
    filepath = "training/model-{:02d}.hdf5".format(int(i))
    model.load_weights(filepath)
    last_layer_weights = model.layers[2].get_weights()[0]
    last_layer_bias = model.layers[2].get_weights()[1]

    CYT_weight_values = []
    for input_index in range(3):
        kth_node_weight = last_layer_weights[input_index][0]
        CYT_weight_values.append(kth_node_weight)
    CYT_weight_values.append(last_layer_bias[0])
    CYT_weight_all_iterations.append(CYT_weight_values)

fig1 = plot_weights_for_all_iterations(CYT_weight_all_iterations, 4, num_epochs)
fig1.suptitle('CYT weights all iterations', fontsize=16)
plt.savefig('CYT weights all iterations.png')

#get all iteraions of training ERR
error_all_iteration = []
for i in range(1, num_epochs + 1):
    filepath = "training/model-{:02d}.hdf5".format(int(i))
    model.load_weights(filepath)
    ratio_misclassified = get_misclassified_ratio_TA(model, trainX, trainY)
    error_all_iteration.append(ratio_misclassified)

fig7 = plot_error_for_all_iterations(error_all_iteration, num_epochs)
fig7.suptitle('Training Error for CYT', fontsize=16)
plt.savefig('Training Error for CYT.png')

#get all iteraions of testing ERR
error_all_iteration = []
for i in range(1, num_epochs + 1):
    filepath = "training/model-{:02d}.hdf5".format(int(i))
    model.load_weights(filepath)
    ratio_misclassified = get_misclassified_ratio_TA(model, testX, testY)
    error_all_iteration.append(ratio_misclassified)

fig8 = plot_error_for_all_iterations(error_all_iteration, num_epochs)
fig8.suptitle('Testing Error for CYT', fontsize=16)
plt.savefig('Testing Error for CYT.png')

