from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from prob3_utility_funcs import *

dataset = []
with open('data.txt') as f: #parse the data
    for line in f:
        m = line.split()
        row_vec = []
        for each in m:
            row_vec.append(each)
        dataset.append(row_vec)

for each in dataset: #cast string to data values for each sample
    for i in range(0, 8):
        each[i] = (float)(each[i])

training_coefficient_set = []
training_error_set = []
MPG_vec = extract_col(dataset, 0)
training_vec_output = MPG_vec[:200]
test_vec = MPG_vec[201 : len(MPG_vec)]
for i in range(1, 8):
    feature_vec = extract_col(dataset, i)
    feature_vec = feature_vec[:200]
    for j in range(0, 4):
        w = single_feature_regression(feature_vec, training_vec_output, j)
        training_coefficient_set.append(w)

fig = plt.figure()
x_var = extract_col(dataset, 5)
y_var = extract_col(dataset, 0)
plt.plot(x_var, y_var, 'g.', markersize=1)
x = []
coeffs = training_coefficient_set[18]
y = []
for i in range(int(min(x_var)), int(max(x_var))):
    jerk = predicted(coeffs, i, 2)
    x.append(i)
    y.append(jerk)
plt.plot(x, y)
plt.show()
print(training_coefficient_set)