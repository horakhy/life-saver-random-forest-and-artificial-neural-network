from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_log_error
import numpy as np
from load_the_data import (
    given_data_classifier,
    given_data_regressor,
    targeted_labels,
    targeted_gravity,
)
import matplotlib.pyplot as plt

def decision_tree_classifier():
    X_attributes = given_data_classifier.drop(["label"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_labels, test_size=0.2, random_state=1
    )

    ## Criar o modelo do classificador de árvore de decisão
    dtree_model = DecisionTreeClassifier(criterion="gini", max_depth=2)
    dtree_model.fit(X_train, y_train)

    ## Predições
    dtree_predictions = dtree_model.predict(X_test)

    validate = np.isin(dtree_predictions, targeted_labels)

    plot_tree(dtree_model)
    plt.savefig("dtree.png")

    print("Accuracy:", accuracy_score(y_test, dtree_predictions))
    print("validate:", validate.all())
    
    export_graphviz(dtree_model, 
      out_file='Tree.dot', 
      feature_names = X_train.columns,
      rounded = True, proportion = False, 
      precision = 2, filled = True)

def decision_tree_regressor():
    X_attributes = given_data_regressor.drop(["gravity"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_gravity, test_size=0.2, random_state=1
    )

    ## Criar o modelo do classificador de árvore de decisão
    dtree_model = DecisionTreeRegressor(criterion="poisson", max_depth=8)
    dtree_model.fit(X_train, y_train)

    ## Predições
    dtree_predictions = dtree_model.predict(X_test)

    validate = np.isin(dtree_predictions, targeted_gravity)

    plot_tree(dtree_model)
    plt.savefig("dtree.png")

    print(r2_score(y_test, dtree_predictions))
    print(mean_squared_log_error(y_test, dtree_predictions))

    print(y_test[:10])
    print(dtree_predictions[:10])


decision_tree_classifier()
# decision_tree_regressor()
