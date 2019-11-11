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
trainX = dataset[0:1, 1:9]
trainY = dataset[0:1, 9]
trainY = encoding(trainY)

X = np.array(trainX)
Y = np.array(trainY)

model = Sequential()
model.add(Dense(3, input_dim=8, activation='sigmoid',
                kernel_initializer='zeros', bias_initializer='zeros'))
model.add(Dense(3, input_dim=8, activation='sigmoid',
                kernel_initializer='zeros', bias_initializer='zeros'))
model.add(Dense(10, activation='sigmoid',
                kernel_initializer='zeros', bias_initializer='zeros'))

layer1 = []
x = np.array([[1, 0, 0] for _ in range(3)]) #weights
y = np.array([1, 0, 0]) #array of biases
layer1.append(x)
layer1.append(y)
model.layers[1].set_weights(layer1) #loaded_model.layer[0] being the layer


layer2 = []
x = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(3)]) #weights
y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #array of biases
layer2.append(x)
layer2.append(y)
model.layers[2].set_weights(layer2) #loaded_model.layer[0] being the layer

print("The initialized weight is: \n", model.get_weights())
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mse'])
model.fit(X, Y, epochs=1)
print("The updated weight is: \n", model.get_weights())

last_layer_weights = model.layers[2].get_weights()[0]
last_layer_bias = model.layers[2].get_weights()[1]
w = []
for input_index in range(3):
    kth_node_weight = last_layer_weights[input_index][0]
    w.append(kth_node_weight)
print("Last layer weight is ", w)
print("Last layer bias is ", last_layer_bias[0])

last_layer_weights = model.layers[1].get_weights()[0]
last_layer_bias = model.layers[1].get_weights()[1]
w = []
for input_index in range(3):
    kth_node_weight = last_layer_weights[input_index][0]
    w.append(kth_node_weight)

print("Second last layer weight is ", w)
print("Second last layer bias is ", last_layer_bias[0])