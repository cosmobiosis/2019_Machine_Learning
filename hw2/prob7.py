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

trainX = np.array(trainX)
trainY = np.array(trainY)
# checkpoint_path = "training/"
# jerk = os.path.dirname(checkpoint_path)

lowest_layers = 1
lowest_nodes = 3
lowest_error = 9999
lowest_model = Sequential()
testing_error_set = []
for num_hidden_layers in range(1, 4):
    for num_hidden_nodes in ([3, 6, 9, 12]):
        model = vary_train_ANN_RELU(trainX, trainY, num_epochs, num_hidden_layers, num_hidden_nodes)
        #error = model.evaluate(np.asarray(testX), np.asarray(testY), verbose=1)[1]
        error = get_misclassified_ratio(model, testX, testY)
        testing_error_set.append(error)
        if error < lowest_error:
            lowest_layers = num_hidden_layers
            lowest_nodes = num_hidden_nodes
            lowest_model = model

model = lowest_model
#get lowest model compiled
error_all_iteration = []
for i in range(1, num_epochs + 1):
    filepath = "training/model-{:02d}.hdf5".format(int(i))
    model.load_weights(filepath)
    error = get_misclassified_ratio_TA(model, trainX, trainY)
    error_all_iteration.append(error)
    #result = model.evaluate(np.asarray(trainX), np.asarray(trainY), verbose=1)
    #acc = result[1]
    #error_all_iteration.append(1-acc)

fig7 = plot_error_for_all_iterations(error_all_iteration, num_epochs)
fig7.suptitle('lowest training error', fontsize=16)
plt.savefig('lowest training error.png')

#get all iteraions of testing ERR
error_all_iteration = []
for i in range(1, num_epochs + 1):
    filepath = "training/model-{:02d}.hdf5".format(int(i))
    model.load_weights(filepath)
    error = get_misclassified_ratio_TA(model, testX, testY)
    error_all_iteration.append(error)
    #result = model.evaluate(np.asarray(testX), np.asarray(testY), verbose=1)
    #acc = result[1]
    #error_all_iteration.append(1-acc)

fig8 = plot_error_for_all_iterations(error_all_iteration, num_epochs)
fig8.suptitle('lowest testing error', fontsize=16)
plt.savefig('lowest testing error.png')

index = 0
for num_hidden_layers in range(1, 4):
    for num_hidden_nodes in ([3, 6, 9, 12]):
        print("Layer ", num_hidden_layers, " Node ", num_hidden_nodes, "is \n",
              testing_error_set[index])
        index += 1