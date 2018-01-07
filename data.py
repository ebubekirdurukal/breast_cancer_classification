import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from timeit import default_timer as timer

data=pd.read_csv(r'C:\Users\DELL\PycharmProjects\breast_cancer_classification\data.csv')
col = data.columns       # .columns gives columns names in data
y = data.diagnosis                          # M or B
list = ['Unnamed: 32','id','diagnosis']
x = data.drop(list,axis = 1 )
scaler = MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler()
#x_transformed = min_max_scaler.fit_transform(x)
x_transformed = scaler.fit_transform(x)
#print x_transformed


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
X_tra_train, X_tra_test, y_tra_train, y_tra_test = train_test_split(x_transformed, y, test_size=0.25, random_state=42)




print len(x)
print len(x_transformed)



logreg = linear_model.LogisticRegression(C=1e5)
knn = KNeighborsClassifier(n_neighbors=3)
model = GaussianNB()
tree = tree.DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=100)
support = svm.SVC()
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)


print ("using all the features")
print('\n' * 1)

start = timer()
scores = cross_val_score(clf, x_transformed, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for scaled features (Random forest): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(clf, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for non-scaled values (Random forest): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)

#print precision_recall_fscore_support(y_train, y_pred, average='macro')
#print precision_recall_fscore_support(y_tra_train, y_pred_tra, average='macro')

print ("non scaled features used below since there is no significant difference")
print('\n' * 1)

start = timer()
scores = cross_val_score(logreg, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for logistic regression: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)

start = timer()
scores = cross_val_score(model, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for Gaussian naive bayes: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)

start = timer()
scores = cross_val_score(knn, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for k nearest neighbours: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(tree, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy Decision Trees: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(support, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for support vector machines : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(bdt, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for Adaboost : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)




drop_list1 = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se',
                  'perimeter_se', 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst',
                  'compactness_se', 'concave points_se', 'texture_worst', 'area_worst']
x = x.drop(drop_list1, axis=1)

print('\n' * 2)
print ("using feature selection methods:  ")

start = timer()
scores = cross_val_score(clf, x_transformed, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for scaled features (Random forest): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(clf, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for non scaled values (Random forest): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)

#print precision_recall_fscore_support(y_train, y_pred, average='macro')
#print precision_recall_fscore_support(y_tra_train, y_pred_tra, average='macro')
print ("non scaled features used below since there is no significant difference")
print('\n' * 1)
start = timer()
scores = cross_val_score(logreg, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for logistic regression: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)

start = timer()
scores = cross_val_score(model, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for Gaussian naive bayes: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)

start = timer()
scores = cross_val_score(knn, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for k nearest neighbours: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(tree, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy Decision Trees: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(support, x_transformed, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for support vector machines : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


start = timer()
scores = cross_val_score(bdt, x, y, cv=10)
elapsed_time = timer() - start # in seconds
print("Accuracy for Adaboost : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "time elapsed is :", elapsed_time)


# find best scored 5 features
#select_feature = SelectKBest(chi2, k=5).fit(X_train, y_train)
#print('Score list:', select_feature.scores_)
#print('Feature list:', X_train.columns)

#x_train_2 = select_feature.transform(X_train)
#x_test_2 = select_feature.transform(X_test)
#random forest classifier with n_estimators=10 (default)



#print precision_recall_fscore_support(y_train, y_pred, average='macro')
#print precision_recall_fscore_support(y_tra_train, y_pred_tra, average='macro')
#print accuracy_score(y_test, y_pred)
#target_names = ['class 0', 'class 1']
#print(classification_report(y_test, y_pred, target_names=target_names))
#print  precision_score(y_test, y_pred, average='micro')
#print roc_auc_score(y_test, y_pred)







from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=10,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

#print('Optimal number of features :', rfecv.n_features_)
#print('Best features :', X_train.columns[rfecv.support_])





