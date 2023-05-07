import pickle
import csv
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from artificial_neural_network import plot_data, plot_confusion_matrix
from sklearn.metrics import classification_report, r2_score, mean_squared_error

arquivo_classificador = "treino_sinais_vitais_com_label.txt"
arquivo_regressor = "treino_sinais_vitais_com_label.txt"

classification_classes = {
    1: "Crítico",
    2: "Instável",
    3: "Potencialmente estável",
    4: "Estável" 
}

column_names_without_label = ["id", "pSist", "pDiast", "qPA", "Pulse", "fResp", "gravity"]
column_names_with_label = ["id", "pSist", "pDiast", "qPA", "Pulse", "fResp", "gravity", "label"]

given_data_regressor= pd.read_csv(arquivo_regressor, names=column_names_with_label, header=None)
given_data_classifier= pd.read_csv(arquivo_classificador, names=column_names_with_label, header=None)

given_data_regressor = given_data_regressor.drop(['id', 'pSist', 'pDiast', 'gravity','label'], axis=1)
given_data_classifier = given_data_classifier.drop(['id', 'pSist', 'pDiast','label'], axis=1)

# load model
filename = "regressor-NN.pickle"
loaded_model = pickle.load(open(filename, "rb"))


prediction = loaded_model.predict(given_data_regressor)
print(prediction)

#Insert the metrics
