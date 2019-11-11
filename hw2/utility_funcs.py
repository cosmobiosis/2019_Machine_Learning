import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model
import matplotlib.pyplot as plt

def train_ANN(X, Y, epochs):
    X = np.array(X)
    Y = np.array(Y)
    #checkpoint_path = "training/"
    #jerk = os.path.dirname(checkpoint_path)
    model = Sequential()
    model.add(Dense(3, input_dim=8, activation='sigmoid',
                    kernel_initializer='glorot_uniform', bias_initializer='ones'))
    model.add(Dense(3, activation='sigmoid',
                    kernel_initializer='glorot_uniform', bias_initializer='ones'))
    model.add(Dense(10, activation='sigmoid',
                    kernel_initializer='glorot_uniform', bias_initializer='ones'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    filepath="training/model-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1)
    callbacks_list = [checkpoint]
    model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=1, callbacks=callbacks_list, verbose=0)
    return model


# train ANN using various num of hidden layers and nodes
def vary_train_ANN(X, Y, epochs, num_hidden_layers, num_hidden_nodes):
    X = np.array(X)
    Y = np.array(Y)
    #checkpoint_path = "training/"
    #jerk = os.path.dirname(checkpoint_path)
    model = Sequential()
    for _ in range(num_hidden_layers):
        model.add(Dense(num_hidden_nodes, activation='sigmoid'))
    model.add(Dense(10, init='glorot_uniform', activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=10, verbose=0)
    return model

# train ANN using various num of hidden layers and nodes
def vary_train_ANN_RELU(X, Y, epochs, num_hidden_layers, num_hidden_nodes):
    X = np.array(X)
    Y = np.array(Y)
    #checkpoint_path = "training/"
    #jerk = os.path.dirname(checkpoint_path)
    model = Sequential()
    for _ in range(num_hidden_layers):
        model.add(Dense(num_hidden_nodes, init='glorot_uniform', activation='relu'))
    model.add(Dense(10, init='glorot_uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "training/model-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [checkpoint]
    model.fit(X, Y, validation_split=0.33, epochs=epochs, batch_size=10, callbacks=callbacks_list, verbose=0)
    return model

def plot_weights_for_all_iterations(node_all_weights_all_epochs, num_weights, num_epochs):
    #node_all_output_weights_all_epochs: Matrix with shape (num_epochs , num_output_weights)
    #this records all output weights for all epochs
    fig = plt.figure()
    for i in range(num_weights):
        curNode_weights_all_epochs = []
        epoch_axis = []
        for epoch in range(num_epochs):
            epoch_axis.append(epoch)
            curNode_weights_all_epochs.append(node_all_weights_all_epochs[epoch][i])
        plt.plot(epoch_axis, curNode_weights_all_epochs)

    return fig

def plot_error_for_all_iterations(error_all_epoch, num_epochs):
    fig = plt.figure()
    epoch_axis = []
    for epoch in range(num_epochs):
        epoch_axis.append(epoch)
    plt.plot(epoch_axis, error_all_epoch)

    return fig

def get_misclassified_ratio(model, X, Y):
    result = model.evaluate(np.asarray(X), np.asarray(Y), verbose=1)
    return 1 - result[1]

def get_misclassified_ratio_alternative(model, X, Y):
    predict = np.array(np.argmax(model.predict(X), axis=1))
    true = np.array(np.argmax(Y, axis=1))

    mis_count = 0
    num_true_CYT = np.count_nonzero(true == 0)
    for i in range(len(predict)):
        if true[i] == 0 and predict[i] != 0:
            mis_count += 1

    return mis_count/num_true_CYT

def get_misclassified_ratio_TA(model, X, Y):
    predict = np.array(np.argmax(model.predict(X), axis=1))
    true = np.array(np.argmax(Y, axis=1))

    mis_count = 0
    for i in range(len(predict)):
        if true[i] == 0 and predict[i] != 0:
            mis_count += 1

    return mis_count/len(predict)

def encoding(classes):  #encode the last column to vector representation
    y = []
    for i in range(len(classes)):
        if classes[i] == 'CYT':
            y.append([1,0,0,0,0,0,0,0,0,0])
        if classes[i] == 'NUC':
            y.append([0,1,0,0,0,0,0,0,0,0])
        if classes[i] == 'MIT':
            y.append([0,0,1,0,0,0,0,0,0,0])
        if classes[i] == 'ME3':
            y.append([0,0,0,1,0,0,0,0,0,0])
        if classes[i] == 'ME2':
            y.append([0,0,0,0,1,0,0,0,0,0])
        if classes[i] == 'ME1':
            y.append([0,0,0,0,0,1,0,0,0,0])
        if classes[i] == 'EXC':
            y.append([0,0,0,0,0,0,1,0,0,0])
        if classes[i] == 'VAC':
            y.append([0,0,0,0,0,0,0,1,0,0])
        if classes[i] == 'POX':
            y.append([0,0,0,0,0,0,0,0,1,0])
        if classes[i] == 'ERL':
            y.append([0,0,0,0,0,0,0,0,0,1])
    return y
