from operator import itemgetter
from prob3_utility_funcs import *
import matplotlib.pyplot as plt

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

for each in dataset: #cast string to data values for each sample
    for i in range(0, 8):
        each[i] = (float)(each[i])

sortedData = sorted(dataset, key=itemgetter(0))

low_mpg_class = []
medium_mpg_class = []
high_mpg_class = []
veryhi_mpg_class = []
#Threshold the data
first_threshold = sortedData[(int)(len(sortedData) /4)][0]
second_threshold = sortedData[(int)(len(sortedData) /2)][0]
third_thereshold = sortedData[(int)(len(sortedData)* 3/4)][0]

fig = plt.figure()
plt.rc('font', size=1)  # controls default text sizes
plt.rc('axes', titlesize=2)  # fontsize of the axes title
plt.rc('axes', labelsize=2)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=1)  # fontsize of the tick labels
plt.rc('ytick', labelsize=1)  # fontsize of the tick labels
plt.rc('legend', fontsize=1)  # legend fontsize
for i in range(1, 8):
    for j in range(1, 8):
        linearInd = (i - 1) * 7 + j
        ax = fig.add_subplot(7, 7, linearInd)
        switcher = {
            1: 'cylinders',
            2: 'displacement',
            3: 'horsepower',
            4: 'weight',
            5: 'acceleration',
            6: 'model year',
            7: 'origin'
        }
        xlabel = switcher.get(i)
        ylabel = switcher.get(j)
        ax.set_title(xlabel)
        plt.ylabel(ylabel)

        x_var = extract_col(dataset, i)
        y_var = extract_col(dataset, j)
        if i != j:
            mpg_var = extract_col(dataset, 0)
            for k in range(0, len(dataset)):
                if mpg_var[k] <= first_threshold:
                    ax.plot(x_var[k], y_var[k], 'k.', markersize=1)
                if (mpg_var[k] > first_threshold) and (mpg_var[k] <= second_threshold):
                    ax.plot(x_var[k], y_var[k], 'r.', markersize=1)
                if (mpg_var[k] > second_threshold) and (mpg_var[k] <= third_thereshold):
                    ax.plot(x_var[k], y_var[k], 'y.', markersize=1)
                if mpg_var[k] > third_thereshold:
                    ax.plot(x_var[k], y_var[k], 'g.', markersize=1)
        else:
            ax.hist(x_var, bins='auto')
plt.tight_layout()
plt.savefig('ScatterPlotMatrix.png', dpi=1000)

