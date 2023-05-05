import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_log_error
from load_the_data import given_data_classifier, given_data_regressor, targeted_labels, targeted_gravity

def plot_data(y_test, predict_test):
	df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': predict_test})
	df_temp.head()
	df_temp = df_temp.head(50)
	df_temp.plot(kind='bar',figsize=(10,6))
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.savefig("NN-plot.png")

def neural_network_classifier():
	target_column = ['label'] 

	predictors = list(set(list(given_data_classifier.columns))-set(target_column))
	max_value_predictors= given_data_classifier[predictors].max()

	given_data_classifier[predictors] = given_data_classifier[predictors]/max_value_predictors ## Normalização dos dados
	given_data_classifier.describe().transpose()

	X_train, X_test, y_train, y_test = train_test_split(given_data_classifier, targeted_labels, test_size=0.3, random_state=1)
	print(X_train.shape)
	print(X_test.shape)

	mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=100)
	mlp.fit(X_train,y_train)

	predict_train = mlp.predict(X_train)
	predict_test = mlp.predict(X_test)

	plot_data(y_test, predict_test)

	print(len(y))
	print(len(predict_train))

	print(confusion_matrix(y_test, predict_test))
	print(classification_report(y_test, predict_test))
	print(confusion_matrix(y_test,predict_test))
	print(classification_report(y_test, predict_test))


def neural_network_regressor():
    X_attributes = given_data_regressor.drop(["gravity"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_attributes, targeted_gravity, test_size=0.2, random_state=0
    )
    sc= StandardScaler() # normaliza para retiar o bias do modelo para valores altos.
    scaler = sc.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(120, 50, 20), activation='relu', solver='adam', max_iter=2000, early_stopping=True)
    mlp.fit(X_train_scaled,y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test_scaled)
    validate = np.isin(predict_train, targeted_gravity)
    
    plot_data(y_test, predict_test)
    
    print(r2_score(y_test, predict_test))
    print(mean_squared_log_error(y_test, predict_test))

    print(validate)

neural_network_classifier()