import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def print_cmx(y_true, y_pred):
    label = [i for i in range(51)]
    labels = sorted(list(set(label)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cmx, annot=True, vmax=4, vmin=0, cmap='plasma')
    plt.savefig("./figure/confusionMatrix.png")
    plt.show()
