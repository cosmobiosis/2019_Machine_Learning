#bootstrapping
import pandas as pd
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import numpy as np
from prob2 import *

df = pd.read_excel('ecs171.dataset.xlsx')
df = pd.DataFrame(df)
df.drop(['b4635'], axis=1)
growth_vec = df['GrowthRate']
jerk = df['Strain']
df['Strain'] = [dict([(y,x) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]
jerk = df['Medium']
df['Medium'] = [dict([(y,x) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]
jerk = df['Stress']
df['Stress'] = [dict([(y,x) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]
jerk = df['GenePerturbed']
df['GenePerturbed'] = [dict([(y,x) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]

attributes = df.columns
gene_names = attributes[6:-1]
feature_mat = df[gene_names].fillna(0)

#concatenation
values = pd.concat([feature_mat.iloc[:, :], growth_vec.iloc[:]], axis=1, sort=False)
sample_to_predict = feature_mat.iloc[:, :].mean(axis=0)
alpha = 0.95
lower, upper = get_confidence_interval(values, sample_to_predict, alpha)
print('%.1f%% confidence interval %.3f and %.3f' % (alpha * 100, lower, upper))