from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

df = pd.read_excel('ecs171.dataset.xlsx')
df = pd.DataFrame(df)
df.drop(['b4635'], axis=1)
growth_vec = df['GrowthRate']
jerk = df['Strain']
df['Strain'] = [dict([(y,x+1) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]
jerk = df['Medium']
df['Medium'] = [dict([(y,x+1) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]
jerk = df['Stress']
df['Stress'] = [dict([(y,x+1) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]
jerk = df['GenePerturbed']
df['GenePerturbed'] = [dict([(y,x+1) for x,y in enumerate(sorted(set(jerk)))])[x] for x in jerk]

attributes = df.columns
gene_names = attributes[6:-1]
origin_feature_mat = df[gene_names].fillna(0)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(origin_feature_mat)
origin_feature_mat = scaler.transform(origin_feature_mat)

pca = PCA(n_components=2)
clf = pca.fit(origin_feature_mat)
feature_mat = pca.transform(origin_feature_mat)

fig1 = plt.figure()
plt.scatter(feature_mat[:, 0], feature_mat[:, 1], alpha=1, lw=0.0001)
plt.title("PCA")
plt.show()

tsne = TSNE(n_components=2)
feature_mat = tsne.fit_transform(origin_feature_mat)
fig2 = plt.figure()
plt.scatter(feature_mat[:, 0], feature_mat[:, 1], alpha=1, lw=0.0001)
plt.title("TSNE")
plt.show()
