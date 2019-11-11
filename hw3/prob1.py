import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
df = pd.read_excel('ecs171.dataset.xlsx')
df = pd.DataFrame(df)
df.drop(['b4635'], axis=1)
growth_vec = df['GrowthRate']

attributes = df.columns
gene_names = attributes[6:-1]
feature_mat = df[gene_names].fillna(0)

coefs = []
alphas = [.0001, .0005, .001]
lasso = linear_model.Lasso(normalize=True, max_iter=1e5)

for alpha in alphas:
    lasso.set_params(alpha = alpha)
    lasso.fit(feature_mat, growth_vec)
    print(np.count_nonzero(lasso.coef_ != 0))
    coefs.append(lasso.coef_)

lassocv = linear_model.LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(feature_mat, growth_vec)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(feature_mat, growth_vec)
print(mean_squared_error(growth_vec, lasso.predict(feature_mat)))




