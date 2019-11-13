from sklearn.svm import SVC
from sklearn import linear_model
import pandas as pd
from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier
from prob_utility_funcs import *

df = pd.read_excel('ecs171.dataset.xlsx')
df = pd.DataFrame(df)
df.drop(['b4635'], axis=1)
growth_vec = df['GrowthRate']

attributes = df.columns
gene_names = attributes[6:-1]
origin_feature_mat = df[gene_names].fillna(0)

lasso = linear_model.Lasso(normalize=True, max_iter=1e5)
lasso.set_params(alpha = .001)
lasso.fit(origin_feature_mat, growth_vec)

chosen_feature_names = []
for i in range(len(lasso.coef_)):
    if lasso.coef_[i] != 0:
        chosen_feature_names.append(gene_names[i])
X = origin_feature_mat[chosen_feature_names]

strain = df["Strain"]
medium = df["Medium"]
stress = df["Stress"]
perturb = df["GenePerturbed"]

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
fig1 = get_ROC_10fold_plot(classifier, X, strain, 10)
fig1.suptitle('Strain ROC with 10-fold cross-validated probabilities')
plt.savefig('Strain ROC.png')

fig5 = get_PR_curve_10fold_plot(classifier, X, strain, 10)
fig5.suptitle('Strain PR with 10-fold cross-validated probabilities')
plt.savefig('Strain PR.png')

fig2 = get_ROC_10fold_plot(classifier, X, medium, 18)
fig2.suptitle('Medium ROC with 10-fold cross-validated probabilities')
plt.savefig('Medium ROC.png')

fig6 = get_PR_curve_10fold_plot(classifier, X, medium, 18)
fig6.suptitle('Medium PR with 10-fold cross-validated probabilities')
plt.savefig('Medium PR.png')

fig3 = get_ROC_10fold_plot(classifier, X, stress, 7)
fig3.suptitle('Stress ROC with 10-fold cross-validated probabilities')
plt.savefig('Stress ROC.png')

fig7 = get_PR_curve_10fold_plot(classifier, X, stress, 7)
fig7.suptitle('Stress PR with 10-fold cross-validated probabilities')
plt.savefig('Stress PR.png')

fig4 = get_ROC_10fold_plot(classifier, X, perturb, 10)
fig4.suptitle('Perturb ROC with 10-fold cross-validated probabilities')
plt.savefig('Perturb ROC.png')

fig8 = get_PR_curve_10fold_plot(classifier, X, perturb, 10)
fig8.suptitle('Perturb PR with 10-fold cross-validated probabilities')
plt.savefig('Perturb PR.png')


