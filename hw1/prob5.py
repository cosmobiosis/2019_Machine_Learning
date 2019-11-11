from operator import itemgetter
from prob3_utility_funcs import *
import random

feature_mat = []
MPG_vec = []

def is_valid(row):
    for each in row:
        if each == '?':
            return False
    return True
dataset = []
with open('auto-mpg.data') as f: #parse the data
    for line in f:
        m = line.split()
        if is_valid(m):
            row_vec = []
            for each in m:
                row_vec.append(each)
            dataset.append(row_vec)

random.seed(3)
random.shuffle(dataset) #shuffle the dataset to make the distribution nicer

for each in dataset: #cast string to data values for each sample
    for i in range(0, 8):
        each[i] = (float)(each[i])

for each in dataset:
    MPG_vec.append(each[0])
    feature_mat.append(each[1:8])

for each in feature_mat: #cast string to data values for each sample
    for i in range(0, 7):
        each[i] = (float)(each[i])

training_error_set = []
testing_error_set = []
training_vec_y = MPG_vec[:200]
test_vec_y = MPG_vec[200:]

train_feature_mat = feature_mat[:200]
test_feature_mat = feature_mat[200:]

for power in range(0, 3):
    w = multi_feature_regression(train_feature_mat, training_vec_y, power)

    trainError = getMeanError_Multifeature(train_feature_mat, w, training_vec_y, power)
    training_error_set.append(trainError)

    testError = getMeanError_Multifeature(test_feature_mat, w, test_vec_y, power)
    testing_error_set.append(testError)

for power in range(0, 3):
    print("Training Error for " + str(power) +  "th power polynomial is "
              + str(training_error_set[power]))
    print("Testing Error for " + str(power) + "th power polynomial is "
          + str(testing_error_set[power]))
    print('\n')