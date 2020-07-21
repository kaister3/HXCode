#线性回归一般用于连续值的预测
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 8)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#线性回归
lr = LinearRegression(normalize=True, n_jobs=2)
scores = cross_val_score(lr, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
print(scores.mean())

lr.fit(X_train, y_train)
lr.score(X_test, y_test)

from sklearn.preprocessing import PolynomialFeatures

#多项式回归

for k in range(1, 4):
    lr_featurizer = PolynomialFeatures(degree=k)
    print ('----%d----' % k)

    X_pf_train = lr_featurizer.fit_transform(X_train)
    X_pf_test = lr_featurizer.transform(X_test)

    pf_scores = cross_val_score(lr, X_pf_train, y_train, cv = 10, scoring = 'neg_mean_squared_error')
    print(pf_scores.mean())

    lr.fit(X_pf_train, y_train)
    print (lr.score(X_pf_test, y_test))
    print (lr.score(X_pf_train, y_train))