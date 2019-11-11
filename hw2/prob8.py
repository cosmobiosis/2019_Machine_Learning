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

X = np.array(trainX)
Y = np.array(trainY)
# checkpoint_path = "training/"
# jerk = os.path.dirname(checkpoint_path)
model = Sequential()
model.add(Dense(3, input_dim=8, activation='sigmoid',
                kernel_initializer='glorot_uniform', bias_initializer='ones'))
model.add(Dense(3, activation='sigmoid',
                kernel_initializer='glorot_uniform', bias_initializer='ones'))
model.add(Dense(10, activation='softmax',
                kernel_initializer='glorot_uniform', bias_initializer='ones'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, validation_split=0.33, epochs=num_epochs, batch_size=1, verbose=1)

unknown = np.array([0.5 , 0.49 , 0.52 , 0.2 , 0.55  ,0.03 , 0.50 , 0.39])
test_sequence = []
test_sequence.append(unknown)
test_sequence = np.array(test_sequence)
predict_sequence = model.predict(test_sequence)

output = predict_sequence[0]
print("the unknown sample has the uncertainty of", 1-output)