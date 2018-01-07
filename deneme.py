
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr



data=pd.read_csv(r'C:\Users\DELL\PycharmProjects\breast_cancer_classification\data.csv')
y = data.diagnosis                          # M or B
list = ['Unnamed: 32','id','diagnosis']
x = data.drop(list,axis = 1 )
col = x.columns
benign = data
malignant = data[data.diagnosis == 'M']
benign = data[data.diagnosis == 'B']
col_ben = benign.columns
col_mal = malignant.columns


for i in range(0, len (col)-1):

        if abs(benign[col[i]].mean() - malignant[col[i]].mean())<0.001 :
            print col[i] , "'s has  very similar means for benign and malignant types"




print('\n' * 4)







for i in range(0, len (col)-1):
    for j in range(1,len (col)):
        if x[col[i]].corr(x[col[j]])>0.90   and  col[i]!=col[j] :
            print "The corrolation between " ,col[i], " and ",col[j]," is " , x[col[i]].corr(x[col[j]])

print('\n' * 4)
print ("xxasdxxasdasdasd")

drop_list1 = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se',
                  'perimeter_se', 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst',
                  'compactness_se', 'concave points_se', 'texture_worst', 'area_worst']
x_1 = x.drop(drop_list1, axis=1)

col = x_1.columns

for i in range(0, len (col)-1):
    for j in range(1,len (col)):
        if x_1[col[i]].corr(x_1[col[j]])>0.90   and  col[i]!=col[j] :
            print "The corrolation between " ,col[i], " and ",col[j]," is " , x[col[i]].corr(x[col[j]])





#for i in range(0, 10):
 #   for j in range(1,11):
  #       m = x.columns[i]
   #      n = x.columns[j]

#    print "The corrolation between " ,m, " and ",n," is " ,pearsonr(m, n)
