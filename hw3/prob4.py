from sklearn import linear_model
import pandas as pd
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
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
fig1, mean_auc_strain = get_ROC_kfold_plot(classifier, X, strain, 10, 5)
fig1.suptitle('Strain ROC with 5-fold cross-validated probabilities')
plt.savefig('Strain ROC.png')

fig2, mean_auc_medium = get_ROC_kfold_plot(classifier, X, medium, 18, 5)
fig2.suptitle('Medium ROC with 5-fold cross-validated probabilities')
plt.savefig('Medium ROC.png')

fig3, mean_auc_stress = get_ROC_kfold_plot(classifier, X, stress, 7, 5)
fig3.suptitle('Stress ROC with 5-fold cross-validated probabilities')
plt.savefig('Stress ROC.png')

fig4, mean_auc_perturb = get_ROC_kfold_plot(classifier, X, perturb, 10, 5)
fig4.suptitle('Perturb ROC with 5-fold cross-validated probabilities')
plt.savefig('Perturb ROC.png')

fig5, mean_pr_strain = get_PR_curve_kfold_plot(classifier, X, strain, 10, 5)
fig5.suptitle('Strain PR with 5-fold cross-validated probabilities')
plt.savefig('Strain PR.png')


fig6, mean_pr_medium = get_PR_curve_kfold_plot(classifier, X, medium, 18, 5)
fig6.suptitle('Medium PR with 5-fold cross-validated probabilities')
plt.savefig('Medium PR.png')


fig7, mean_pr_stress = get_PR_curve_kfold_plot(classifier, X, stress, 7, 5)
fig7.suptitle('Stress PR with 5-fold cross-validated probabilities')
plt.savefig('Stress PR.png')


fig8, mean_pr_perturb = get_PR_curve_kfold_plot(classifier, X, perturb, 10, 5)
fig8.suptitle('Perturb PR with 5-fold cross-validated probabilities')
plt.savefig('Perturb PR.png')

print("Mean AUC for Strain is: ", mean_auc_strain)
print("Mean PRA for Strain is: ", mean_pr_strain)

print("Mean AUC for Medium is: ", mean_auc_medium)
print("Mean PRA for Medium is: ", mean_pr_medium)

print("Mean AUC for Stress is: ", mean_auc_stress)
print("Mean PRA for Stress is: ", mean_pr_stress)

print("Mean AUC for Perturb is: ", mean_auc_perturb)
print("Mean PRA for Perturb is: ", mean_pr_perturb)

print("Mean AUC for two individual Medium and Perturb classifiers is ")
print((mean_auc_perturb + mean_auc_medium) / 2)

print("Mean PRA for two individual Medium and Perturb classifiers is ")
print((mean_pr_perturb + mean_pr_medium) / 2)