from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from load_the_data import given_data, expected_data, expected_labels

# print(expected_labels)

## Separar dataSet para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(given_data, expected_labels, test_size=0.7, random_state=0)

## Criar o modelo de árvore de decisão
dtree_model = DecisionTreeRegressor(max_depth=3)
dtree_model.fit(X_train, y_train)

## Predições
dtree_predictions = dtree_model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, dtree_predictions)) * 100000)

plot_tree(dtree_model)

