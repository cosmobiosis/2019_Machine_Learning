#bootstrapping
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import numpy as np

def get_confidence_interval(values, sample_to_predict, alpha):
	n_iterations = 250
	n_size = int(len(values) * 0.30)
	# run bootstrap
	stats = list()
	for i in range(n_iterations):
		# prepare train and test sets
		train = resample(values, n_samples=n_size)
		#test = np.array([x for x in values if x.tolist() not in train.tolist()])
		# fit model
		lasso = linear_model.Lasso(normalize=True, max_iter=1e5, alpha=.0001)
		lasso.fit(train.iloc[:, :-1], train.iloc[:, -1])
		# evaluate model
		predictions = lasso.predict([sample_to_predict])[0]
		predictions = round(predictions, 4)
		stats.append(predictions)
	# plot scores
	plt.hist(stats)
	plt.show()
	# confidence intervals
	p = ((1.0 - alpha) / 2.0) * 100
	lower = max(0.0, np.percentile(stats, p))
	p = (alpha + ((1.0 - alpha) / 2.0)) * 100
	upper = min(1.0, np.percentile(stats, p))
	return lower, upper
