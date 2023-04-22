import csv

import pandas as pd

given_data = []
expected_data = []
classification_classes = {
    1: "Crítico",
    2: "Instável",
    3: "Potencialmente estável",
    4: "Estável" 
}

column_names_without_label = ["id", "pSist", "pDiast", "qPA", "Pulse", "fResp", "gravity"]
column_names_with_label = ["id", "pSist", "pDiast", "qPA", "Pulse", "fResp", "gravity", "label"]

with open('treino_sinais_vitais_sem_label.txt', newline='') as csvfile:
    # Create a reader object
    reader = csv.reader(csvfile, delimiter=',')

    # Initialize an empty list to store the data

    # Iterate over each row in the CSV file and append it to the data list
    for row in reader:
        row = [cell.strip() for cell in row]
        given_data.append(row)

given_data= pd.read_csv('treino_sinais_vitais_sem_label.txt', names=column_names_without_label, header=None)

expected_data= pd.read_csv('treino_sinais_vitais_com_label.txt', names=column_names_with_label, header=None)

expected_labels = expected_data["label"]