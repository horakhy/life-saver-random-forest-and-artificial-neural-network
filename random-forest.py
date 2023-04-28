from sklearn.ensemble import RandomForestClassifier
import numpy as np
from load_the_data import given_data_classifier, given_data_regressor
import matplotlib.pyplot as plt

## Separar dataSet para treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(given_data, expected_labels, test_size=0.7, random_state=0)

# X_attributes = given_data.drop(['label'], axis=1)

## Criar o modelo de árvore de decisão
# dtree_model = RandomForestClassifier(n_estimators=100, criterion='gini')
# dtree_model.fit(X_attributes, targeted_labels)

## Predições
# dtree_predictions = dtree_model.predict(evaluation_data)

# validate = np.isin(dtree_predictions, targeted_labels)

# print(True if validate.all() else False)