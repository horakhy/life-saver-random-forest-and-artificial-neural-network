from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from load_the_data import given_data, evaluation_data, targeted_labels
import matplotlib.pyplot as plt

## Separar dataSet para treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(given_data, expected_labels, test_size=0.7, random_state=0)

X_attributes = given_data.drop(['label'], axis=1)

## Criar o modelo de árvore de decisão
dtree_model = DecisionTreeClassifier(criterion='gini', max_depth=3)
dtree_model.fit(X_attributes, targeted_labels)

## Predições
dtree_predictions = dtree_model.predict(evaluation_data)

validate = np.isin(dtree_predictions, targeted_labels)

plot_tree(dtree_model)
plt.savefig("dtree.png")
