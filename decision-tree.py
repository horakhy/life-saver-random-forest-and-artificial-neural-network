from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from load_the_data import given_data, expected_data


## Separar dataSet para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(given_data, expected_data, test_size=0.3, random_state=1)

## Criar o modelo de árvore de decisão
clf = DecisionTreeClassifier()

## Treinar o modelo
clf = clf.fit(X_train,y_train)

## Fazer previsões
y_pred = clf.predict(X_test)

## Avaliar a acurácia
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
# print("Recall:", metrics.recall_score(y_test, y_pred, average='weighted'))
# print("F1 score:", metrics.f1_score(y_test, y_pred, average='weighted'))


