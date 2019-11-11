import numpy as np
import matplotlib.pyplot as plt
from prob3_utility_funcs import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import random
from sklearn.metrics import precision_score

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

sortedData = sorted(dataset, key=itemgetter(0))
first_threshold = sortedData[(int)(len(sortedData) /4)][0]
second_threshold = sortedData[(int)(len(sortedData) /2)][0]
third_thereshold = sortedData[(int)(len(sortedData)* 3/4)][0]

ylabel = []
for each in MPG_vec:    #append labels to MPG
    if each <= (float)(first_threshold):
        ylabel.append(0)
    elif each > (float)(first_threshold) and each <= (float)(second_threshold):
        ylabel.append(1)
    elif each > (float)(second_threshold) and each <= (float)(third_thereshold):
        ylabel.append(2)
    else:
        ylabel.append(3)

train_y = ylabel[:200]
test_y = ylabel[200:]

train_feature_mat = feature_mat[:200]
test_feature_mat = feature_mat[200:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)

scaler.fit(train_feature_mat)
train_feature_mat = scaler.transform(train_feature_mat)
scaler.fit(test_feature_mat)
test_feature_mat = scaler.transform(test_feature_mat)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)
clf.fit(train_feature_mat, train_y)

print("training precision is")
print(classification_report(train_y, clf.predict(train_feature_mat), [0,1,2,3]))
print("testing precision is ")
print(classification_report(test_y, clf.predict(test_feature_mat), [0,1,2,3]))