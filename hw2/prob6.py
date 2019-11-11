from utility_funcs import *
import matplotlib.pyplot as plt
import numpy as np
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
trainY = dataset[:, 9]
trainY = encoding(trainY)

num_epochs = 150
model = train_ANN(trainX, trainY, num_epochs)

unknown = np.array([0.5 , 0.49 , 0.52 , 0.2 , 0.55  ,0.03 , 0.50 , 0.39])
test_sequence = []
test_sequence.append(unknown)
test_sequence = np.array(test_sequence)
predict_sequence = model.predict(test_sequence)

output = predict_sequence[0]
print("the unknown sample has the output of", output)

max_Ind = 0
max = -1
for i in range(len(output)):
    if output[i] > max:
        max = output[i]
        max_Ind = i
name_list = ["CYT", "NUC", "MIT", "ME3", "ME2", "ME1", "EXC", "VAC", "POX", "ERL"]
print("the unknown sample is likely to be", name_list[max_Ind])



