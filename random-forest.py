from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from load_the_data import given_data_classifier, given_data_regressor, targeted_gravity, targeted_labels
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score, mean_squared_log_error

def random_forest_classifier():
    X_attributes = given_data_classifier.drop(["label"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_labels, test_size=0.3, random_state=1
    )

    ## Criar o modelo do classificador de árvore de decisão
    dtree_model = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=2)
    dtree_model.fit(X_train, y_train)

    ## Predições
    dtree_predictions = dtree_model.predict(X_test)

    validate = np.isin(dtree_predictions, targeted_labels)

    print("Accuracy:", accuracy_score(y_test, dtree_predictions))
    print("validate:", validate.all())

    print(validate)

    # print(y_test[:10])
    # print(dtree_predictions[:10])


def random_forest_regressor():
    X_attributes = given_data_regressor.drop(["gravity"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_gravity, test_size=0.2, random_state=1
    )

    ## Criar o modelo do classificador de árvore de decisão
    dtree_model = RandomForestRegressor(n_estimators=25, criterion="poisson")
    dtree_model.fit(X_train, y_train)

    ## Predições
    dtree_predictions = dtree_model.predict(X_test)

    validate = np.isin(dtree_predictions, targeted_gravity)

    print(r2_score(y_test, dtree_predictions))
    print(mean_squared_log_error(y_test, dtree_predictions))

    print(validate)
    # print(dtree_predictions[:10])

random_forest_classifier()