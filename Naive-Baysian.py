# import requirement
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
# load data
bcd = datasets.load_breast_cancer()
x = bcd.data
y = bcd.target
# divide data and fit model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=123)

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
print(gnb.score(x_test, y_test)*100)