import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt
from sklearn import preprocessing
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

data=pd.read_csv(r'C:\Users\DELL\PycharmProjects\breast_cancer_classification\data.csv')
y=data.diagnosis
list = ['Unnamed: 32','id','diagnosis']
x = data.drop(list,axis = 1 )
#print x

min_max_scaler = preprocessing.MinMaxScaler()
x_transformed = min_max_scaler.fit_transform(data.drop(list,axis = 1 ))
#print x_transformed


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
X_tra_train, X_tra_test, y_tra_train, y_tra_test = train_test_split(x_transformed, y, test_size=0.33, random_state=42)

print len(X_tra_test)
print len(X_tra_train)

clf = RandomForestClassifier(n_estimators=100,max_depth=2, random_state=0)


clf.fit(X_train, y_train)
clf.fit(X_tra_train, y_tra_train)

y_pred =clf.predict(X_test)
y_pred_tra =clf.predict(X_tra_test)


#print precision_recall_fscore_support(y_test, y_pred, average='macro')
#print precision_recall_fscore_support(y_tra_test, y_pred_tra, average='macro')


#sns.jointplot(x.loc[:,'concavity_worst'], x.loc[:,'concave points_worst'], kind="regg", color="#ce1414")
#plt.show()

col = data.columns       # .columns gives columns names in data
print(col)