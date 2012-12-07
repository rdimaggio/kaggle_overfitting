import numpy as np
import csv as csv
import re
from scipy.stats import spearmanr
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import precision_score
from sklearn.preprocessing import normalize
from sklearn.utils import check_arrays
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

def wmae(y_true, y_pred, weights):
    y_true, y_pred = check_arrays(y_true, y_pred)
    return (1/np.sum(weights)) * (np.sum(weights*(y_pred - y_true)))

def replace_NA(row):
  for each in range(len(row)):
    row[each] = row[each].replace("NA", "0")
    row[each] = "0" if row[each]=="" else row[each]
  return row

def string_find_code(item, sub):
    for i in range(len(sub)):
        if item.find(sub[i]) >= 0:
            return float(i+1) 
    return 0.

def name_length(name):
    try:
        index = name.index(" (")
        return float(len(name[:index-1].split(" ")))
    except:
        return float(len(name.split(" ")))

def convert_currency(row, items=[]):
  for each in items:
    row[each] = row[each].replace("$","").replace(",","")
  return row

def dual_cross_val_score(estimator1, estimator2, X, y, score_func, train, test, verbose, ratio):
    """Inner loop for cross validation"""

    estimator1.fit(X[train], y[train])
    estimator2.fit(X[train], y[train])

    guess = ratio*estimator1.predict(X[test]) + (1-ratio)*estimator2.predict(X[test])
    guess[ guess < 0.5 ] = 0.
    guess[ guess >= 0.5 ] = 1.
    score = score_func(y[test], guess)
    
    if verbose > 1:
      print("score: %f" % score)
    return score

def Bootstrap_cv(estimator1, estimator2, X, y, score_func, cv=None, n_jobs=1,
                    verbose=0, ratio=.5):
    X, y = cross_validation.check_arrays(X, y, sparse_format='csr')
    cv = cross_validation.check_cv(cv, X, y, classifier=cross_validation.is_classifier(estimator1))
    if score_func is None:
      if not hasattr(estimator1, 'score') or not hasattr(estimator2, 'score'):
        raise TypeError(
            "If no score_func is specified, the estimator passed "
            "should have a 'score' method. The estimator %s "
            "does not." % estimator)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    scores = cross_validation.Parallel(n_jobs=n_jobs, verbose=verbose)(
                cross_validation.delayed(dual_cross_val_score)(
                  cross_validation.clone(estimator1), cross_validation.clone(estimator2), 
                  X, y, score_func, train, test, verbose, ratio)
                for train, test in cv)
    return np.array(scores)


csv_file_object = csv.reader(open('overfitting.csv', 'rb')) #Load in the training csv file
header = csv_file_object.next() #Skip the fist line as it is a header
train_data=[] #Create a variable called 'train_data'
for row in csv_file_object: #Skip through each row in the csv file
    train_data.append(row) #adding each row to the data variable
train_data = np.array(train_data) #Then convert from a list to an array

train_data = train_data[0:250]
train_data = np.delete(train_data, [0,1,2,4],1)
test_data, train_data = cross_validation.train_test_split(train_data, test_size=0.8, random_state=0)

rho, pval = spearmanr(train_data)
open_file_object = csv.writer(open("correlation.csv", "wb"))
for row in rho:
    open_file_object.writerow(row)

open_file_object = csv.writer(open("pval.csv", "wb"))
for row in pval:
    open_file_object.writerow(row)

pca = PCA(n_components=50)
pca.fit(train_data[0::,1::], train_data[0::,0])
new_train_data = pca.fit_transform(train_data[0::,1::], train_data[0::,0])
new_test_data = pca.transform(test_data[0::,1::])
est = SVR(kernel="linear")
est.fit(new_train_data[0::,1::], new_train_data[0::,0])
print est.score(new_test_data[0::,1::], new_test_data[0::,0])

estimator = SVR(kernel="linear")
selector = RFE(estimator, 50, step=1)
selector = selector.fit(train_data[0::,1::], train_data[0::,0])
print selector.n_features_
print selector.score(test_data[0::,1::], test_data[0::,0])



"""
#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
train_data[train_data[0::,3]=='male',3] = 1
train_data[train_data[0::,3]=='female',3] = 0
#embark c=0, s=1, q=2
train_data[train_data[0::,10] =='C',10] = 0
train_data[train_data[0::,10] =='S',10] = 1
train_data[train_data[0::,10] =='Q',10] = 2

#I need to fill in the gaps of the data and make it complete.
#So where there is no price, I will assume price on median of that class
#Where there is no age I will give median of all ages

#All the ages with no data make the median of the data
train_data[train_data[0::,4] == '',4] = np.median(train_data[train_data[0::,4]\
                                           != '',4].astype(np.float))
#All missing ebmbarks just make them embark from most common place
train_data[train_data[0::,10] == '',10] = np.round(np.mean(train_data[train_data[0::,10]\
                                                   != '',10].astype(np.float)))
#Add three columns for new data
train_data = np.append(train_data, np.zeros((train_data.shape[0],20)),1)

for row in train_data:
  name = row[2].lower()
  cabin = row[9].lower()
  #Name has parenthesis
  row[11] = string_find_code(name, [" ("])
  #Name has mr., miss., mrs., master
  row[12] = string_find_code(name, ["miss.", "mrs.", "mr.", "master"])
  #Name length
  row[13] = name_length(name)
  #Cabin letter
  row[22] = string_find_code(cabin, ["a", "b", "c", "d", "e", "f", "g"])
  #Cabin number
  number = re.findall(r"\d+", cabin)
  row[23] = float(number[0]) if number else 0


train_data[train_data[0::,10] ==0,14] = 1
train_data[train_data[0::,10] ==1,15] = 1
train_data[train_data[0::,10] ==2,16] = 1
train_data[train_data[0::,12] =='1.0',17] = 1
train_data[train_data[0::,12] =='2.0',18] = 1
train_data[train_data[0::,12] =='3.0',19] = 1
train_data[train_data[0::,12] =='4.0',20] = 1
train_data[train_data[0::,9] !='',21] = 1
train_data[train_data[0::,9] =='',21] = 0
train_data[train_data[0::,22] =='1.0',24] = 1
train_data[train_data[0::,22] =='2.0',25] = 1
train_data[train_data[0::,22] =='3.0',26] = 1
train_data[train_data[0::,22] =='4.0',27] = 1
train_data[train_data[0::,22] =='5.0',28] = 1
train_data[train_data[0::,22] =='6.0',29] = 1
train_data[train_data[0::,22] =='7.0',30] = 1

open_file_object = csv.writer(open("full_train.csv", "wb"))
for row in train_data:
    open_file_object.writerow(row)

train_data = np.delete(train_data,[2,7,9,10],1) #remove the name data, cabin and ticket
#I need to do the same with the test data now so that the columns are in the same
#as the training data
train_data = np.delete(train_data,[3,9,10,11,12,20,25,26],1)

test_file_object = csv.reader(open('test.csv', 'rb')) #Load in the test csv file
header = test_file_object.next() #Skip the fist line as it is a header
test_data=[] #Creat a variable called 'test_data'
for row in test_file_object: #Skip through each row in the csv file
    test_data.append(row) #adding each row to the data variable
test_data = np.array(test_data) #Then convert from a list to an array

#I need to convert all strings to integer classifiers:
#Male = 1, female = 0:
test_data[test_data[0::,2]=='male',2] = 1
test_data[test_data[0::,2]=='female',2] = 0
#ebark c=0, s=1, q=2
test_data[test_data[0::,9] =='C',9] = 0 #Note this is not ideal, in more complex 3 is not 3 tmes better than 1 than 2 is 2 times better than 1
test_data[test_data[0::,9] =='S',9] = 1
test_data[test_data[0::,9] =='Q',9] = 2

#All the ages with no data make the median of the data
test_data[test_data[0::,3] == '',3] = np.median(test_data[test_data[0::,3]\
                                           != '',3].astype(np.float))
#All missing ebmbarks just make them embark from most common place
test_data[test_data[0::,9] == '',9] = np.round(np.mean(test_data[test_data[0::,9]\
                                                   != '',9].astype(np.float)))
#All the missing prices assume median of their respectice class
for i in xrange(np.size(test_data[0::,0])):
    if test_data[i,7] == '':
        test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
                                             (test_data[0::,0] == test_data[i,0])\
            ,7].astype(np.float))

#Add three columns for new data
test_data = np.append(test_data, np.zeros((test_data.shape[0],20)),1)

for row in test_data:
  name = row[1].lower()
  cabin = row[8].lower()
  #Name has parenthesis
  row[10] = string_find_code(name, [" ("])
  #Name has mr., miss., mrs., master
  row[11] = string_find_code(name, ["miss.", "mrs.", "mr.", "master"])
  #Name length
  row[12] = name_length(name)
  #Cabin letter
  row[21] = string_find_code(cabin, ["a", "b", "c", "d", "e", "f", "g"])
  #Cabin number
  number = re.findall(r"\d+", cabin)
  row[22] = float(number[0]) if number else 0

test_data[test_data[0::,9] ==0,13] = 1
test_data[test_data[0::,9] ==1,14] = 1
test_data[test_data[0::,9] ==2,15] = 1
test_data[test_data[0::,11] =='1.0',16] = 1
test_data[test_data[0::,11] =='2.0',17] = 1
test_data[test_data[0::,11] =='3.0',18] = 1
test_data[test_data[0::,11] =='4.0',19] = 1
test_data[test_data[0::,8] !='',20] = 1
test_data[test_data[0::,8] =='',20] = 0
test_data[test_data[0::,21] =='1.0',23] = 1
test_data[test_data[0::,21] =='2.0',24] = 1
test_data[test_data[0::,21] =='3.0',25] = 1
test_data[test_data[0::,21] =='4.0',26] = 1
test_data[test_data[0::,21] =='5.0',27] = 1
test_data[test_data[0::,21] =='6.0',28] = 1
test_data[test_data[0::,21] =='7.0',29] = 1


test_data = np.delete(test_data,[1,6,8,9],1) #remove the name data, cabin and ticket
test_data = np.delete(test_data,[2,8,9,10,11,19,24,25],1)


train_data = train_data.astype(np.float)
test_data = test_data.astype(np.float)

rho, pval = spearmanr(train_data)
open_file_object = csv.writer(open("correlation.csv", "wb"))
for row in rho:
    open_file_object.writerow(row)
open_file_object = csv.writer(open("pval.csv", "wb"))
for row in pval:
    open_file_object.writerow(row)


#The data is now ready to go. So lets train then test!
print 'Training'
forest = RandomForestClassifier(n_estimators=50,  min_samples_split=3, \
  min_samples_leaf=2, compute_importances=True, n_jobs=-1)
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

extra_forest = ExtraTreesClassifier(n_estimators=10, max_depth=None, \
  min_samples_split=3, min_samples_leaf=2, compute_importances=True, n_jobs=-1)
extra_forest = extra_forest.fit(train_data[0::,1::], train_data[0::,0])

logit = LogisticRegression()
logit = logit.fit(train_data[0::,1::], train_data[0::,0])

gnb = GaussianNB()
bs = cross_validation.Bootstrap(train_data.shape[0], n_bootstraps=10, train_size=.99, random_state=0)

print "Scoring"
scores = cross_validation.cross_val_score(forest, train_data[0::,1::], train_data[0::,0], cv=bs)
print "RF Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
extra_scores = cross_validation.cross_val_score(extra_forest, train_data[0::,1::], train_data[0::,0], cv=10)
print "EF Accuracy: %0.2f (+/- %0.2f)" % (extra_scores.mean(), extra_scores.std() / 2)
#Normalize data
#train_data = normalize(train_data)
#test_data = normalize(test_data)
logit_scores = cross_validation.cross_val_score(logit, train_data[0::,1::], train_data[0::,0], cv=10)
print "Logit Accuracy: %0.2f (+/- %0.2f)" % (logit_scores.mean(), logit_scores.std() / 2)
gnb_scores = cross_validation.cross_val_score(gnb, train_data[0::,1::], train_data[0::,0], cv=10)
print "GNB Accuracy: %0.2f (+/- %0.2f)" % (gnb_scores.mean(), gnb_scores.std() / 2)
bs_scores = Bootstrap_cv(forest, logit, train_data[0::,1::], train_data[0::,0], score_func=precision_score, cv=10, ratio=.2)
print "RF+Logit Accuracy: %0.2f (+/- %0.2f)" % (bs_scores.mean(), bs_scores.std() / 2)
bse_scores = Bootstrap_cv(extra_forest, logit, train_data[0::,1::], train_data[0::,0], score_func=precision_score, cv=10, ratio=.8)
print "EF+Logit Accuracy: %0.2f (+/- %0.2f)" % (bse_scores.mean(), bse_scores.std() / 2)

print 'Predicting'
score = []
ratio = .2
estimators = 20
train_size = .7
#output = ratio*forest.predict(test_data) + (1-ratio)*logit.predict(test_data)
output = extra_forest.predict(test_data)
"""
"""
#Get bootstrapped data
bs = cross_validation.Bootstrap(train_data.shape[0], n_bootstraps=estimators, train_size=train_size, random_state=0)
cv = cross_validation.check_cv(bs, train_data[0::,1::], train_data[0::,0], classifier=cross_validation.is_classifier(extra_forest))
for train, test in cv:
  #Create training data
  X = train_data[0::,1::]
  y = train_data[0::,0]
  #Create estimator
  est = cross_validation.clone(extra_forest)
  est.fit(X[train], y[train])
  #print est.feature_importances_
  score.append(est.score(X[test], y[test]))

#Format output
score = np.array(score)

#Score
print score
print score.mean()
"""
"""
print 'Outputting'
open_file_object = csv.writer(open("ensemble.csv", "wb"))
test_file_object = csv.reader(open('test.csv', 'rU')) #Load in the csv file
test_file_object.next()
i = 0
for row in test_file_object:
    row.insert(0,output[i].astype(np.uint8))
    open_file_object.writerow(row)
    i += 1
"""