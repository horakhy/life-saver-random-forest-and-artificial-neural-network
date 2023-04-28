import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from load_the_data import given_data, evaluation_data, targeted_labels

target_column = ['label'] 
predictors = list(set(list(given_data.columns))-set(target_column))
given_data[predictors] = given_data[predictors]/given_data[predictors].max()
given_data.describe().transpose()

X = given_data[predictors].values
y = given_data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(given_data, targeted_labels, test_size=0.7, random_state=0)
print(X_train.shape); print(X_test.shape)


mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=5000)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
