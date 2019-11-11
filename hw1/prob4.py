from operator import itemgetter
from prob3_utility_funcs import *
import random

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

training_coefficient_set = []
training_error_set = []
testing_error_set = []
MPG_vec = extract_col(dataset, 0)
training_vec_y = MPG_vec[:200]
test_vec_y = MPG_vec[200:]
for i in range(1, 8):
    feature_vec = extract_col(dataset, i)
    train_feature_vec = feature_vec[:200]
    test_feature_vec = feature_vec[200:]
    for j in range(0, 4):
        w = single_feature_regression(train_feature_vec, training_vec_y, j)
        training_coefficient_set.append(w)

        trainError = getMeanError(train_feature_vec, w, training_vec_y, j)
        training_error_set.append(trainError)

        testError = getMeanError(test_feature_vec, w, test_vec_y, j)
        testing_error_set.append(testError)


for i in range(0, 7):   #plot the testing data
    x_vec = extract_col(dataset, i+1)
    x_vec = x_vec[200:]
    four_coeffs = []
    for j in range(0, 4):
        training_index = i * 4 + j
        four_coeffs.append(training_coefficient_set[training_index])
    fig = plot_Polynomial_Regression(x_vec, four_coeffs, test_vec_y)
    switcher = {
        0: 'cylinders',
        1: 'displacement',
        2: 'horsepower',
        3: 'weight',
        4: 'acceleration',
        5: 'model year',
        6: 'origin'
    }
    name = str(i + 1) + ' ' + switcher.get(i) + ' Poly.png'
    fig.suptitle(switcher.get(i) + ' 0to3 poly-order test', fontsize=16)
    plt.savefig(name, dpi=1000)

    for j in range(0, 4):
        switcher = {
            0: 'cylinders',
            1: 'displacement',
            2: 'horsepower',
            3: 'weight',
            4: 'acceleration',
            5: 'model year',
            6: 'origin'
        }
        training_index = i * 4 + j
        print("Training Error for " + switcher.get(i) + " in its " + str(j) + "th polynomial is "
              + str(training_error_set[training_index]))
        print("Testing Error for " + switcher.get(i) + " in its " + str(j) + "th polynomial is "
              + str(testing_error_set[training_index]))
    print('\n')


#print(training_coefficient_set)
#[array(19.74), array([36.53996815, -2.84745223]), array([38.98732991, -3.74799478,  0.07551619]),
# array([-100.99320791,   73.54577396,  -13.54272618,    0.76816377]), array(19.74), array([29.82829059, -0.04507273]),
# array([ 3.48234469e+01, -9.94367015e-02,  1.13531982e-04]), array([ 6.30046178e-03,  4.21390210e-01, -2.04592235e-03,  2.66703822e-06]), array(19.74), array([32.39080089, -0.10941233]), array([ 4.45796494e+01, -3.19999326e-01,  7.97273903e-04]), array([ 1.92443885e-02,  7.82830017e-01, -7.54306894e-03,  1.95287047e-05]), array(19.74), array([ 3.77772685e+01, -5.66349491e-03]), array([ 1.13890495e-05,  1.79369390e-02, -3.42101960e-06]), array([ 1.79801136e-12,  3.89496778e-09,  6.65248308e-06, -1.34646324e-09]), array(19.74), array([3.8734168 , 1.05516946]), array([-11.79001797,   3.1756541 ,  -0.06921089]), array([ 3.95866113e+01, -7.57309427e+00,  6.52885100e-01, -1.56159936e-02]), array(19.74), array([-23.76911384,   0.59695567]), array([ 8.28559488e+02, -2.28010186e+01,  1.60470436e-01]), array([ 4.70533171e-01,  1.14293186e+01, -3.10975513e-01,  2.16340694e-03]), array(19.74), array([11.78583757,  5.50461068]), array([ 1.37374236, 19.17503452, -3.59987177]), array([ 8.31030389,  6.45800505,  3.33668976, -1.15609359])]
