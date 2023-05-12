import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

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
given_data_classifier = given_data_classifier.drop(['id', 'pSist', 'pDiast', 'gravity'], axis=1)

targeted_gravity = given_data_regressor["gravity"]
targeted_labels = given_data_classifier["label"]


def plot_importance(clf, png_name):
    plt.clf()
    # Creating importances_df dataframe
    importances_df = pd.DataFrame({"feature_names" : clf.feature_names_in_, 
                                "importances" : clf.feature_importances_})
                                
    # Plotting bar chart, g is from graph
    g = sns.barplot(x=importances_df["feature_names"], 
                    y=importances_df["importances"])
    g.set_title("Feature importances", fontsize=14) 
    plt.savefig(f'importance_{png_name}.png')
    plt.close()

def plot_confusion_matrix(model, x_test, y_test, png_name):
    fig=ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=["1","2","3","4"])
    fig.figure_.suptitle("Confusion Matrix")
    plt.savefig(f'confusion_matrix_{png_name}.png')
    plt.show()
    plt.close()