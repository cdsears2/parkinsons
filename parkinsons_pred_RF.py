
"""UCIPD.ipynb
Prediction of Parkinson's disease in patients from vocal recording data using https://archive.ics.uci.edu/dataset/174/parkinsons.
"""

import pandas as pd
import numpy as np

#Exploratory data analysis
df = pd.read_csv('parkinsons.data')
print(df.head(),df.shape)

#data preparation and creating train/val/test sets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics

x = df.drop(columns=['name','status']).values
y = df['status'].values

#data scaling
scaler = MinMaxScaler((0,1))
x = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.25,random_state=7)

#training with random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,random_state=0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

#measuring performance
"""Measuring performance using ROC-AUC"""
from sklearn.metrics import confusion_matrix,accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,RocCurveDisplay
import matplotlib.pyplot as plt


fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()
auc = roc_auc_score(y_test, y_pred)
print(auc)