from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree,export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
import numpy as np
import pickle
from load_the_data import (
    given_data_classifier,
    given_data_regressor,
    plot_importance,
    targeted_labels,
    targeted_gravity,
    plot_confusion_matrix
)
import matplotlib.pyplot as plt
import random

best_max_depth = 0
best_random_state = 0
best_accuracy_score = 0

def decision_tree_classifier(max_depth):
    X_attributes = given_data_classifier.drop(["label"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_labels, test_size=0.4, random_state=1
    )
    
    dtree_model = DecisionTreeClassifier(criterion="gini", max_depth=12)
    dtree_model.fit(X_train, y_train)

    ## Predições
    dtree_predictions = dtree_model.predict(X_test)
    # filename = "regressor-Tree.pickle"
    # pickle.dump(dtree_model, open(filename, "wb"))

    # plot_tree(dtree_model)
    # plt.savefig("dtree-classifier.png")

    print(classification_report(y_test, dtree_predictions))
    # plot_importance(dtree_model, "tree-classifier.png")
    # plot_confusion_matrix(dtree_model, X_test, y_test, "decision-tree-classifier")
    
    global best_accuracy_score, best_max_depth, best_random_state
    if(accuracy_score(y_test, dtree_predictions) > best_accuracy_score):    
        best_accuracy_score = accuracy_score(y_test, dtree_predictions)
        best_max_depth = max_depth

    # export_graphviz(dtree_model, 
    #   out_file='Tree.dot', 
    #   feature_names = X_train.columns,
    #   rounded = True, proportion = False, 
    #   precision = 2, filled = True)

def decision_tree_regressor():
    X_attributes = given_data_regressor.drop(["gravity"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_gravity, test_size=0.3, random_state=1
    )
    
    ## Criar o modelo do classificador de árvore de decisão
    dtree_model = DecisionTreeRegressor(criterion="poisson", max_depth=16)
    dtree_model.fit(X_train, y_train)

    ## Predições
    dtree_predictions = dtree_model.predict(X_test)
    filename = "regressor-Tree.pickle"
    pickle.dump(dtree_model, open(filename, "wb"))

    # plot_tree(dtree_model)
    # plt.savefig("dtree-regressor.png")

    print(r2_score(y_test, dtree_predictions))
    print(mean_squared_error(y_test, dtree_predictions))


for i in range(1):
    decision_tree_classifier(random.randint(1, 25))
    print("Iteração: ", i)

# decision_tree_regressor()


# print("Best max_depth: ", best_max_depth)
# print("Best random_state: ", best_random_state)
# print("Best accuracy_score: ", best_accuracy_score)