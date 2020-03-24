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

medium = df["Medium"].values
perturb = df["GenePerturbed"].values

composite_labels = []
for i in range(len(medium)):
    join_ = [medium[i], perturb[i]]
    s = " "
    concaten_string = s.join(join_)
    composite_labels.append(concaten_string)

print(dict([(y,x+1) for x,y in enumerate(sorted(set(composite_labels)))]))
compos_ = [dict([(y,x+1) for x,y in enumerate(sorted(set(composite_labels)))])[x] for x in composite_labels]
print(compos_)

most_label = np.bincount(compos_).argmax()
correct = np.count_nonzero(compos_== most_label)
print("Baseline Most Frequent Label is", most_label)
print("Total #number of True Positive is", correct)
print("Precision for baseline is", correct/len(compos_))

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
fig1, mean_auc = get_ROC_kfold_plot(classifier, X, compos_, 26, 10)
fig1.suptitle('Composite ROC with 10-fold cross-validated probabilities')
plt.savefig('Composite ROC.png')
print(mean_auc)

fig1, mean_pr = get_PR_curve_kfold_plot(classifier, X, compos_, 26, 10)
fig1.suptitle('Composite PR with 10-fold cross-validated probabilities')
plt.savefig('Composite PR.png')
print(mean_pr)