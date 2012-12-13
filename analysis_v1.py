import numpy as np
import csv as csv
from datetime import datetime
#from scipy.stats import spearmanr
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import auc_score
#from sklearn.preprocessing import normalize
from sklearn.utils import check_arrays
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.grid_search import GridSearchCV

start = datetime.now()
"""
train on the first 250 items in the target_practice column
then score based upon the remaining ~20k items
"""


def wmae(y_true, y_pred, weights):
    y_true, y_pred = check_arrays(y_true, y_pred)
    return (1 / np.sum(weights)) * (np.sum(weights * (y_pred - y_true)))


def replace_NA(row):
    for each in range(len(row)):
        row[each] = row[each].replace("NA", "0")
        row[each] = "0" if row[each] == "" else row[each]
    return row


def string_find_code(item, sub):
    for i in range(len(sub)):
        if item.find(sub[i]) >= 0:
            return float(i + 1)
    return 0.


def name_length(name):
    try:
        index = name.index(" (")
        return float(len(name[:index - 1].split(" ")))
    except:
        return float(len(name.split(" ")))


def convert_currency(row, items=[]):
    for each in items:
        row[each] = row[each].replace("$", "").replace(",", "")
    return row


def dual_cross_val_score(estimator1, estimator2, X, y, score_func,
                         train, test, verbose, ratio):
    """Inner loop for cross validation"""

    estimator1.fit(X[train], y[train])
    estimator2.fit(X[train], y[train])

    guess = ratio * estimator1.predict(X[test]) + (1 - ratio) * \
        estimator2.predict(X[test])
    guess[guess < 0.5] = 0.
    guess[guess >= 0.5] = 1.
    score = score_func(y[test], guess)

    if verbose > 1:
        print("score: %f" % score)
    return score


def Bootstrap_cv(estimator1, estimator2, X, y, score_func, cv=None, n_jobs=1,
                 verbose=0, ratio=.5):
    X, y = cross_validation.check_arrays(X, y, sparse_format='csr')
    cv = cross_validation.check_cv(cv, X, y,
                                   classifier=
                                   cross_validation.is_classifier(estimator1))
    if score_func is None:
        if not hasattr(estimator1, 'score') or \
                not hasattr(estimator2, 'score'):
            raise TypeError(
                "If no score_func is specified, the estimator passed "
                "should have a 'score' method. The estimator %s "
                "does not." % estimator1)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    scores = \
        cross_validation.Parallel(
            n_jobs=n_jobs, verbose=verbose)(
                cross_validation.delayed(
                    dual_cross_val_score)
                (cross_validation.clone(estimator1),
                 cross_validation.clone(estimator2),
                 X, y, score_func, train, test, verbose, ratio)
                for train, test in cv)
    return np.array(scores)

# load data
csv_file_object = csv.reader(open('overfitting.csv', 'rb'))
header = csv_file_object.next()
all_data = []
for row in csv_file_object:
    all_data.append(row)
all_data = np.array(all_data)
all_data = all_data.astype(np.float)

cutoff = 250
components = 113

# create each data set to use
# all data
all_y_practice = all_data[0::, 2]
all_y_leaderboard = all_data[0::, 3]
all_x = np.delete(all_data, [0, 1, 2, 3, 4], 1)

# train data
train_data = all_data[:cutoff]
train_y_practice = train_data[0::, 2]
train_y_leaderboard = train_data[0::, 3]
train_x = np.delete(train_data, [0, 1, 2, 3, 4], 1)

# test data
test_data = all_data[cutoff:]
entries = test_data[0::, 0]
test_y_practice = test_data[0::, 2]
test_x = np.delete(test_data, [0, 1, 2, 3, 4], 1)

# feature selection
pca = PCA(n_components=.9)
pca.fit(all_x)
train_x_reduced = pca.transform(train_x)
test_x_reduced = pca.transform(test_x)
print pca.components_.shape

"""
estimator = SVR(kernel="linear")
rfe = RFE(estimator=estimator, n_features_to_select=components)
rfe.fit(train_x, train_y_practice)
train_x_reduced = rfe.transform(train_x)
test_x_reduced = rfe.transform(test_x)
"""

print 'Predicting'
#linearSVC regression
parameters = {'penalty': ('l1', 'l2'),
              'tol': (.00001, .0001, .001, .01),
              'C': (10, 100, 500, 1000)}
lsvc = LinearSVC(dual=False)
clf = GridSearchCV(lsvc, parameters, cv=40, score_func=auc_score)
clf.fit(train_x_reduced, train_y_practice)
print "LinearSVC"
print clf.best_estimator_
print clf.best_params_
print clf.best_score_

lsvc_new = LinearSVC(C=100, class_weight=None, dual=False,
                     fit_intercept=True, intercept_scaling=1,
                     penalty='l2', tol=0.001)
lsvc_new.fit(train_x_reduced, train_y_practice)
print lsvc_new.score(test_x_reduced, test_y_practice)


#logistic regression
parameters = {'penalty': ('l1', 'l2'),
              'tol': (.00001, .0001, .001, .01),
              'C': (10, 100, 500, 1000)}
logit = LogisticRegression()
clf = GridSearchCV(logit, parameters, cv=40, score_func=auc_score)
clf.fit(train_x_reduced, train_y_practice)
print "Logit"
print clf.best_estimator_
print clf.best_params_
print clf.best_score_

logit_new = LogisticRegression(C=100, class_weight=None, dual=False,
                                   fit_intercept=True, intercept_scaling=1,
                                   penalty='l1', tol=0.01)
logit_new.fit(train_x_reduced, train_y_practice)
print logit_new.score(test_x_reduced, test_y_practice)


"""
#multinomial nb
parameters = {'alpha': (0, .1, .5, 1.0),
              'fit_prior': (True, False)}
multi_nb = MultinomialNB()
clf = GridSearchCV(multi_nb, parameters, cv=40)
clf.fit(train_x_reduced, train_y_practice)
print "Multinomial NB"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_

multi_nb_new = MultinomialNB(alpha=0, fit_prior=False)
multi_nb_new.fit(train_x_reduced, train_y_practice)
print multi_nb_new.score(test_x_reduced, test_y_practice)
"""
"""
#ElasticNet
parameters = {'alpha': (.1, .25, .5, 1, 2),
              'normalize': (True, False),
              'fit_intercept': (True, False),
              'positive': (True, False)}
enet = ElasticNet()
clf = GridSearchCV(enet, parameters, cv=20)
clf.fit(train_x_reduced, train_y_practice)
print "ElasticNet"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_

enet_new = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=False,
                          max_iter=1000, normalize=True, positive=True,
                          precompute='auto', rho=0.5, tol=0.0001,
                          warm_start=False)
enet_new.fit(train_x_reduced, train_y_practice)
print enet_new.score(test_x_reduced, test_y_practice)
"""
"""
parameters = {'n_estimators': (10, 20, 50),
              'min_samples_split': (1, 2, 4),
              'min_samples_leaf': (1, 2, 4)}
forest = RandomForestClassifier(compute_importances=True, n_jobs=-1)
clf = GridSearchCV(forest, parameters, cv=10, score_func=overfit_score)
clf.fit(train_x_reduced, train_y_practice)
print "RandomForest"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
"""
"""
parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
              'degree': (1, 2, 3, 4),
              'gamma': (0.0, .1, .5)}
svclass = SVC(probability=True, C=.01)
clf = GridSearchCV(svclass, parameters, cv=10)
clf.fit(train_x_reduced, train_y_practice)
print "SVC"
print clf.best_estimator_
print clf.best_score_
print clf.best_params_

svc_new = SVC(probability=True, C=.01, kernel='linear', gamma=0.0,
                  degree=1)
svc_new.fit(train_x_reduced, train_y_practice)
print svc_new.score(test_x_reduced, test_y_practice)
"""

print 'Predicting'
logit_new = LogisticRegression(C=10, class_weight=None, dual=False,
                                   fit_intercept=True, intercept_scaling=1,
                                   penalty='l1', tol=0.0001)
logit_new.fit(train_x_reduced, train_y_leaderboard)
output = logit_new.predict(test_x_reduced)

print 'Outputting'
open_file_object = csv.writer(open(
                              "simple" + str(datetime.now().isoformat()) +
                              ".csv", "wb"))
open_file_object.writerow(['case_id', 'Target_Leaderboard'])
i = 0
for row in entries:
    open_file_object.writerow([row, output[i].astype(np.uint8)])
    i += 1

print 'Done'
print datetime.now() - start
