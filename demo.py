close()

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.decomposition import PCA

data = np.loadtxt('wine.data', delimiter=',')

y, x = np.hsplit(data,np.array([1]))
y = y.reshape(-1)

xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(
    x, y, test_size=0.3, random_state=0)

res = np.zeros(14)
res[0] = 0.3

for n_dim in range(2, 10):
    print n_dim
    pca = PCA(n_components=n_dim).fit(xtrain, ytrain)
    small_xtrain = pca.transform(xtrain)
    small_xtest = pca.transform(xtest)
    small_x = pca.transform(x)

    est = SVC(kernel='linear')
    # est.fit(small_xtrain, ytrain)
    # print est.score(small_xtest, ytest)

    Cs = np.logspace(-20, 1, 10)
    # gammas = np.arange(3)
    clf = GridSearchCV(estimator=est,
                       param_grid=dict(C=Cs),
                       n_jobs=-1,
                       verbose=-1)
    clf.fit(small_xtrain, ytrain)

    tmp = clf.score(small_xtest, ytest)
    res[n_dim] = tmp
    print tmp

if n_dim == 2:
    h = 1
    x_min, x_max = small_x[:, 0].min() - 1, small_x[:, 0].max() + 1
    y_min, y_max = small_x[:, 1].min() - 1, small_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    contourf(xx, yy, Z, cmap=cm.summer)
    axis('off')

    # Plot also the training points
    scatter(small_x[:, 0], small_x[:, 1], c=y, cmap=cm.ocean)
