from operator import itemgetter
from prob3_utility_funcs import *

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

for i in range(0, (int)(len(sortedData) /4)):
    low_mpg_class.append(sortedData[i])
for i in range((int)(len(sortedData) /4), (int)(len(sortedData) /2)):
    medium_mpg_class.append(sortedData[i])
for i in range((int)(len(sortedData)/2), (int)(len(sortedData)* 3/4)):
    high_mpg_class.append(sortedData[i])
for i in range((int)(len(sortedData)* 3/4), (int)(len(sortedData))):
    veryhi_mpg_class.append(sortedData[i])

print("LOW MPG Threshold is between", low_mpg_class[0][0], "and", first_threshold)
print("MEDIUM MPG Threshold is between", first_threshold, "and", second_threshold)
print("HIGH MPG Threshold is between", second_threshold, "and", third_thereshold)
print("VERY HIGH MPG Threshold is between", third_thereshold, "and", veryhi_mpg_class[len(veryhi_mpg_class) - 1][0])

