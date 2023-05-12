import matplotlib.pyplot as plt
import pandas as pd 
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error,ConfusionMatrixDisplay, accuracy_score
from load_the_data import given_data_classifier, given_data_regressor, targeted_labels, targeted_gravity, plot_confusion_matrix

def plot_data(y_test, predict_test):
	df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': predict_test})
	df_temp.head()
	df_temp = df_temp.head(50)
	df_temp.plot(kind='bar',figsize=(10,6))
	plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
	plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.savefig("NN-plot.png")


def neural_network_classifier():

	X_train, X_test, y_train, y_test = train_test_split(given_data_classifier, targeted_labels, test_size=0.4, random_state=1)

	mlp = MLPClassifier(hidden_layer_sizes=(147, 93, 87), activation='relu', learning_rate="adaptive", max_iter=1000)
	mlp.fit(X_train, y_train)

	filename = "classifier-NN.pickle"
	pickle.dump(mlp, open(filename, "wb"))

	predict_test = mlp.predict(X_test)

	plot_data(y_test, predict_test)
	print(confusion_matrix(y_test, predict_test))
	print(classification_report(y_test, predict_test))
	plot_confusion_matrix(mlp, X_test, y_test, "classifier")

def neural_network_regressor():
	X_attributes = given_data_regressor.drop(["gravity"], axis=1)

	X_train, X_test, y_train, y_test = train_test_split(
		X_attributes, targeted_gravity, test_size=0.4, random_state=1
	)
	sc = StandardScaler() # normaliza para retirar o bias do modelo para valores altos.
	scaler = sc.fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	mlp = MLPRegressor(hidden_layer_sizes=(120, 60, 20), activation='relu', learning_rate="adaptive", max_iter=1000)

	mlp.fit(X_train_scaled, y_train)

	predict_test = mlp.predict(X_test_scaled)

	filename = "regressor-NN.pickle"
	pickle.dump(mlp, open(filename, "wb"))
	
	# load model
	#loaded_model = pickle.load(open(filename, "rb"))

	print(r2_score(y_test, predict_test))
	print(mean_squared_error(y_test, predict_test))

neural_network_classifier()
# neural_network_regressor()
