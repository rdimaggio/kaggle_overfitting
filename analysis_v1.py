import numpy as np
import csv as csv
#from scipy.stats import spearmanr
from sklearn import cross_validation
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import precision_score
#from sklearn.preprocessing import normalize
from sklearn.utils import check_arrays
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
#from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

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
                "does not." % estimator)
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
test_y_practice = test_data[0::, 2]
test_x = np.delete(test_data, [0, 1, 2, 3, 4], 1)

# feature selection
pca = PCA(n_components=55)
pca.fit(all_data[0::, 1::], all_data[0::, 0])
print pca.explained_variance_ratio_
train_x_reduced = pca.transform(train_x)
print train_x_reduced.shape

print "PCA"
print pca.components_.shape
print pca.components_
print pca.explained_variance_ratio_.shape
print pca.explained_variance_ratio_


"""
estimator = LogisticRegression()
selector = RFE(estimator, 50, step=1)
selector = selector.fit(train_data[0::, 1::], train_data[0::, 0])
print selector.n_features_
"""

#logistic regression
#parameters = {'penalty':('l1','l2'), 'fit_intercept':(True, False),
#'C':(.25,.5, .75,1, 5,10,100)}
#ElasticNet
parameters = {'alpha': (0, .5, 1, 2, 5),
              'normalize': (True, False),
              'fit_intercept': (True, False),
              'positive': (True, False)}


"""print 'Predicting'
logit = ElasticNet()
clf = GridSearchCV(logit, parameters, cv=20)
clf.fit(train_data, y)
print clf
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
#print clf.score(test_data, test_data[0::, 0])"""

"""
#The data is now ready to go. So lets train then test!
print 'Training'
forest = RandomForestClassifier(n_estimators=100,  min_samples_split=1, \
  min_samples_leaf=1, compute_importances=True, n_jobs=-1)
#forest = forest.fit(train_data[0::,1::], train_data[0::,0])

extra_forest = ExtraTreesClassifier(n_estimators=100, max_depth=None, \
  min_samples_split=3, min_samples_leaf=2, compute_importances=True, n_jobs=-1)
#extra_forest = extra_forest.fit(train_data[0::,1::], train_data[0::,0])

logit = LogisticRegression(penalty='l2', C=.25, fit_intercept=False)
#logit = logit.fit(train_data[0::,1::], train_data[0::,0])

svreg = SVR()
linreg = LinearRegression()
glmnet = ElasticNet()

gnb = GaussianNB()
#bs = cross_validation.Bootstrap(train_data.shape[0], n_bootstraps=10,
                                 train_size=.99, random_state=0)


print "Scoring"
#scores = cross_validation.cross_val_score(forest, train_data, y, cv=10)
#print "RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
extra_scores = cross_validation.cross_val_score(extra_forest, train_data,
                                                y, cv=10)
print "EF Accuracy: %0.2f (+/- %0.2f)" % (extra_scores.mean(),
                                          extra_scores.std() / 2)
logit_scores = cross_validation.cross_val_score(logit, train_data, y, cv=10)
print "Logit Accuracy: %0.2f (+/- %0.2f)" % (logit_scores.mean(),
                                             logit_scores.std() / 2)
gnb_scores = cross_validation.cross_val_score(gnb, train_data, y, cv=10)
print "GNB Accuracy: %0.2f (+/- %0.2f)" % (gnb_scores.mean(),
                                           gnb_scores.std() / 2)
svreg_scores = cross_validation.cross_val_score(svreg, train_data, y, cv=10)
print "SVR Accuracy: %0.2f (+/- %0.2f)" % (svreg_scores.mean(),
                                           svreg_scores.std() / 2)
glm_scores = cross_validation.cross_val_score(glmnet, train_data, y, cv=10)
print "GLMNET Accuracy: %0.2f (+/- %0.2f)" % (glm_scores.mean(),
                                              glm_scores.std() / 2)
#linreg_scores = cross_validation.cross_val_score(linreg, train_data, y, cv=10)
#print "LinearReg Accuracy: %0.2f (+/- %0.2f)" % (linreg_scores.mean(),
                                                  linreg_scores.std() / 2)


print 'Predicting'
logit = LogisticRegression(penalty='l2', C=.25, fit_intercept=False)
logit = LogisticRegression()
logit = logit.fit(train_data, y)
output = logit.predict(eval_data)

print 'Outputting'
open_file_object = csv.writer(open("simple.csv", "wb"))
open_file_object.writerow(['case_id', 'Target_Leaderboard'])
i=0
for row in entries:
    open_file_object.writerow([row, output[i].astype(np.uint8)])
    i += 1
"""
print 'Done'
