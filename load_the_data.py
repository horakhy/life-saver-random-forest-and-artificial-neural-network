import csv

given_data = []
expected_data = []


with open('treino_sinais_vitais_sem_label.txt', newline='') as csvfile:
    # Create a reader object
    reader = csv.reader(csvfile, delimiter=',')

    # Initialize an empty list to store the data

    # Iterate over each row in the CSV file and append it to the data list
    for row in reader:
        row = [cell.strip() for cell in row]
        given_data.append(row)

# read from com_label
with open('treino_sinais_vitais_com_label.txt', newline='') as csvfile:
    # Create a reader object
    reader = csv.reader(csvfile, delimiter=',')

    # Initialize an empty list to store the data

    # Iterate over each row in the CSV file and append it to the data list
    for row in reader:
        row = [cell.strip() for cell in row]
        expected_data.append(row)
