from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from prob_utility_funcs import *

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
pca_feature_mat = pca.transform(origin_feature_mat)

tsne = TSNE(n_components=2)
tsne_feature_mat = tsne.fit_transform(origin_feature_mat)

strain = df["Strain"]
medium = df["Medium"]
stress = df["Stress"]
perturb = df["GenePerturbed"]

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

_, mean_auc_strain = get_ROC_kfold_plot(classifier, pca_feature_mat, strain, 10, 5)
_, mean_pr_strain = get_PR_curve_kfold_plot(classifier, pca_feature_mat, strain, 10, 5)

_, mean_auc_strain_tsne = get_ROC_kfold_plot(classifier, tsne_feature_mat, strain, 10, 5)
_, mean_pr_strain_tsne = get_PR_curve_kfold_plot(classifier, tsne_feature_mat, strain, 10, 5)



_, mean_auc_medium = get_ROC_kfold_plot(classifier, pca_feature_mat, medium, 18, 5)
_, mean_pr_medium = get_PR_curve_kfold_plot(classifier, pca_feature_mat, medium, 18, 5)

_, mean_auc_medium_tsne = get_ROC_kfold_plot(classifier, tsne_feature_mat, medium, 18, 5)
_, mean_pr_medium_tsne = get_PR_curve_kfold_plot(classifier, tsne_feature_mat, medium, 18, 5)


_, mean_auc_stress = get_ROC_kfold_plot(classifier, pca_feature_mat, stress, 7, 5)
_, mean_pr_stress = get_PR_curve_kfold_plot(classifier, pca_feature_mat, stress, 7, 5)

_, mean_auc_stress_tsne = get_ROC_kfold_plot(classifier, tsne_feature_mat, stress, 7, 5)
_, mean_pr_stress_tsne = get_PR_curve_kfold_plot(classifier, tsne_feature_mat, stress, 7, 5)


_, mean_auc_perturb = get_ROC_kfold_plot(classifier, pca_feature_mat, perturb, 10, 5)
_, mean_pr_perturb = get_PR_curve_kfold_plot(classifier, pca_feature_mat, perturb, 10, 5)

_, mean_auc_perturb_tsne = get_ROC_kfold_plot(classifier, tsne_feature_mat, perturb, 10, 5)
_, mean_pr_perturb_tsne = get_PR_curve_kfold_plot(classifier, tsne_feature_mat, perturb, 10, 5)


print("Strain:")
print("Strain PCA AUC:", mean_auc_strain)
print("Strain PCA AUPRC:", mean_pr_strain)
print("Strain TSNE AUC:", mean_auc_strain_tsne)
print("Strain TSNE AUPRC:", mean_pr_strain_tsne)
print("Medium:")
print("Medium PCA AUC:", mean_auc_medium)
print("Medium PCA AUPRC:", mean_pr_medium)
print("Medium TSNE AUC:", mean_auc_medium_tsne)
print("Medium TSNE AUPRC:", mean_pr_medium_tsne)
print("Stress:")
print("Stress PCA AUC:", mean_auc_stress)
print("Stress PCA AUPRC:", mean_pr_stress)
print("Stress TSNE AUC:", mean_auc_stress_tsne)
print("Stress TSNE AUPRC:", mean_pr_stress_tsne)
print("Perturb:")
print("Perturb PCA AUC:", mean_auc_perturb)
print("Perturb PCA AUPRC:", mean_pr_perturb)
print("Perturb TSNE AUC:", mean_auc_perturb_tsne)
print("Perturb TSNE AUPRC:", mean_pr_perturb_tsne)