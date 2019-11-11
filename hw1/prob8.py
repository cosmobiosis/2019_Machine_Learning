feature_to_be_predict = [4.0, 400.0, 150.0, 3500.0, 8.0, 81, 1]

from operator import itemgetter
from prob3_utility_funcs import *
from sklearn.linear_model import LogisticRegression
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

polyBasis = 2
x = []
for i in range(0, len(feature_mat)):  # for i examples
    temp_row = []
    temp_row.append(1)
    for feature_index in range(0, 7):
        for power in range(1, polyBasis + 1):
            temp_row.append(pow(feature_mat[i][feature_index], power))
    x.append(temp_row)
x = np.matrix(x)
w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.transpose(), x)), x.transpose()), MPG_vec)
w = np.squeeze(np.asarray(w))  # change the coefficients from 2D to 1D list

poly_feature_to_predict = []
poly_feature_to_predict.append(1)
for feature_index in range(0, 7):
    for power in range(1, polyBasis + 1):
        poly_feature_to_predict.append(pow(feature_to_be_predict[feature_index], power))


print("the predicted MPG is", np.dot(w, poly_feature_to_predict))

#lOGISTIC REGRESSION
sortedData = sorted(dataset, key=itemgetter(0))
first_threshold = sortedData[(int)(len(sortedData) /4)][0]
second_threshold = sortedData[(int)(len(sortedData) /2)][0]
third_thereshold = sortedData[(int)(len(sortedData)* 3/4)][0]
ylabel = []
for each in MPG_vec:    #append labels to MPG
    if each <= (float)(first_threshold):
        ylabel.append([0])
    elif each > (float)(first_threshold) and each <= (float)(second_threshold):
        ylabel.append([1])
    elif each > (float)(second_threshold) and each <= (float)(third_thereshold):
        ylabel.append([2])
    else:
        ylabel.append([3])

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=100)
clf.fit(feature_mat, ylabel)
feature_to_be_predict = np.matrix(feature_to_be_predict)
print("The label of this car is:", clf.predict(feature_to_be_predict))