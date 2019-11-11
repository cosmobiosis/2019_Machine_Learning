from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np

#prob3 OLS estimator for single feature of nth-order polynomial
def single_feature_regression(feature_vec, MPG_vec, polyBasis):
    x = []
    y = np.matrix(MPG_vec).transpose()
    for i in range(0, len(feature_vec)):
        temp_row = []
        for power in range(0, polyBasis + 1):
            temp_row.append(pow(feature_vec[i], power))
        x.append(temp_row)
    x = np.matrix(x)
    #w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.transpose(), x)), x.transpose()), y)
    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.transpose(), x)), x.transpose()), y)
    w = np.squeeze(np.asarray(w))  # change the coefficients from 2D to 1D list
    return w

#predict y with single feature
def predicted(w, x, polyBasis): #calculate up to the third polynomial: y = w0 + w1x + w2x^2 + w3    x^3
    x_vec = []
    for j in range(0, polyBasis + 1):
        x_vec.append(pow(x, j))
    return np.dot(w, x_vec)

#get meanError for a single feature
def getMeanError(x, w, y, polyBasis):
    error = 0
    counter = 0
    for test_index in range(0, len(y)):
        observed_val = y[test_index]
        test_x = x[test_index]
        predicted_val = predicted(w, test_x, polyBasis)
        error += pow((predicted_val - observed_val), 2)
        counter += 1
    return error / counter

#Multi feature utility collection
#OLS estimator for multi feature of nth-order polynomial
def multi_feature_regression(feature_mat, MPG_vec, polyBasis):
    x = []
    y = MPG_vec
    for i in range(0, len(feature_mat)):    #for i examples
        temp_row = []
        temp_row.append(1)
        for feature_index in range(0, 7):
            for power in range(1, polyBasis + 1):
                temp_row.append(pow(feature_mat[i][feature_index], power))
        x.append(temp_row)
    x = np.matrix(x)
    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.transpose(), x)), x.transpose()), y)
    w = np.squeeze(np.asarray(w))  # change the coefficients from 2D to 1D list
    return w

#predict y with multi features
def predicted_Multifeature(w, x, polyBasis):
    x_vec = []
    x_vec.append(1)
    for feature_index in range(0, 7):
        for power in range(1, polyBasis + 1):
            x_vec.append(pow(x[feature_index], power))
    return np.dot(w, x_vec)

#get meanError with multi features
def getMeanError_Multifeature(x, w, y, polyBasis):
    error = 0
    counter = 0
    for test_index in range(0, len(y)):
        observed_val = y[test_index]
        test_x = x[test_index]
        predicted_val = predicted_Multifeature(w, test_x, polyBasis)
        error += pow((predicted_val - observed_val), 2)
        counter += 1
    return error / counter

def plot_Polynomial_Regression(x_vec, four_coeffs, MPG_vec):    #plot the polynomial regression graph
    fig = plt.figure()
    plt.plot(x_vec, MPG_vec, 'g.', markersize=1)
    for polyBasis in range(0, 4):
        x = []
        y = []
        for i in range(int(min(x_vec)), int(max(x_vec))):
            predicted_val = predicted(four_coeffs[polyBasis], i, polyBasis)
            x.append(i)
            y.append(predicted_val)
        plt.plot(x, y)
    return fig

def extract_col(dataset, i):    #extract i_th column of the dataset
    col = []
    for each in dataset:
        col.append(each[i])
    return col
