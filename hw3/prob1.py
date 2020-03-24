import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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


alphas = [0.0000005, .000001, .00005, .0001, .0005, .0005, .001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9]
lasso = linear_model.Lasso(normalize=True, max_iter=1e5)

for alpha in alphas:
    lasso.set_params(alpha = alpha)
    kf = KFold(n_splits=5)
    error_vec = []
    for train_index, test_index in kf.split(feature_mat):
        X = np.array(feature_mat)
        y = np.array(growth_vec)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_test)
        MSE = np.square(np.subtract(y_test, y_pred)).mean() #code from scratch MSE
        error_vec.append(MSE)
    print("alpha", alpha, "has following mean MSE(5 Fold) of:", np.mean(error_vec), "and #features of ", np.count_nonzero(lasso.coef_ != 0))

'''
coefs = []
alphas = [.000001, .00001, .0001, .0005, .001, 0.01]
parameters = {'alpha' : alphas}
lasso = linear_model.Lasso(normalize=True, max_iter=1e5)

clf = GridSearchCV(lasso, parameters, cv=5)
clf.fit(feature_mat, growth_vec)
print(clf.best_params_)
'''
