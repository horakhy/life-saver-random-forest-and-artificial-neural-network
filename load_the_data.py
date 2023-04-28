import csv

import pandas as pd

classification_classes = {
    1: "Crítico",
    2: "Instável",
    3: "Potencialmente estável",
    4: "Estável" 
}

column_names_without_label = ["id", "pSist", "pDiast", "qPA", "Pulse", "fResp", "gravity"]
column_names_with_label = ["id", "pSist", "pDiast", "qPA", "Pulse", "fResp", "gravity", "label"]

given_data_regressor= pd.read_csv('treino_sinais_vitais_com_label.txt', names=column_names_with_label, header=None)
given_data_classifier= pd.read_csv('treino_sinais_vitais_com_label.txt', names=column_names_with_label, header=None)

given_data_regressor = given_data_regressor.drop(['id', 'pSist', 'pDiast', 'label'], axis=1)
given_data_classifier = given_data_classifier.drop(['id', 'pSist', 'pDiast'], axis=1)



targeted_gravity = given_data_regressor["gravity"]
targeted_labels = given_data_classifier["label"]