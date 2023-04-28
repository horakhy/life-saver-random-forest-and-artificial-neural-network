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
from load_the_data import given_data_classifier, targeted_labels

target_column = ['label'] 

# X_attributes = given_data_classifier.drop(['label'], axis=1)

predictors = list(set(list(given_data_classifier.columns))-set(target_column))
max_value_predictors= given_data_classifier[predictors].max()


given_data_classifier[predictors] = given_data_classifier[predictors]/max_value_predictors ## Normalização dos dados
given_data_classifier.describe().transpose()

X = given_data_classifier[predictors].values
y = given_data_classifier[target_column].values

X_train, X_test, y_train, y_test = train_test_split(given_data_classifier, targeted_labels, test_size=0.3, random_state=1)
print(X_train.shape)
print(X_test.shape)


mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=100)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)


print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train,predict_train))
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
