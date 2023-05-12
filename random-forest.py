from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle
from sklearn.model_selection import train_test_split
from load_the_data import given_data_classifier, given_data_regressor, targeted_gravity, targeted_labels, plot_confusion_matrix
from sklearn.metrics import classification_report, r2_score, mean_squared_error, accuracy_score

def random_forest_classifier():
    X_attributes = given_data_classifier.drop(["label"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_labels, test_size=0.4, random_state=1
    )

    ## Criar o modelo do classificador de árvore de decisão
    dtree_model = RandomForestClassifier(n_estimators=53, criterion="gini", max_depth=14)
    dtree_model.fit(X_train, y_train)
    
    ## Predições
    dtree_predictions = dtree_model.predict(X_test)
    # filename = "classifier-RandF.pickle"
    # pickle.dump(dtree_model, open(filename, "wb"))

    print(classification_report(y_test, dtree_predictions))
    # plot_confusion_matrix(dtree_model, X_test, y_test, "random-forest-classifier")
    
def random_forest_regressor():
    X_attributes = given_data_regressor.drop(["gravity"], axis=1)

    ## Separar dataSet para treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_gravity, test_size=0.3, random_state=1
    )

    ## Criar o modelo do classificador de árvore de decisão
    dtree_model = RandomForestRegressor(n_estimators=25, max_depth=12, criterion="poisson")
    dtree_model.fit(X_train, y_train)

    ## Predições
    dtree_predictions = dtree_model.predict(X_test)
    
    filename = "regressor-RandF.pickle"
    pickle.dump(dtree_model, open(filename, "wb"))

    print(r2_score(y_test, dtree_predictions))
    print(mean_squared_error(y_test, dtree_predictions))


random_forest_classifier()
# random_forest_regressor()