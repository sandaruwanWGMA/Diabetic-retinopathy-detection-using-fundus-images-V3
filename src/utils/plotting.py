import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_loss(losses, title="Loss Over Time", save_path=None):
    """
    Plots the loss function over time.

    Parameters:
        losses (dict): A dictionary where keys are loss function names and values are lists of loss values over epochs.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(10, 6))
    for loss_name, loss_values in losses.items():
        plt.plot(loss_values, label=loss_name)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_losses_to_file(losses, file_path):
    """
    Saves the loss values to a file.

    Parameters:
        losses (dict): A dictionary where keys are loss function names and values are lists of loss values over epochs.
        file_path (str): Path to save the file.
    """
    with open(file_path, "w") as f:
        for loss_name, loss_values in losses.items():
            f.write(f"{loss_name}:\n")
            f.write(", ".join(map(str, loss_values)) + "\n")


def save_classification_report(y_true, y_pred, filename="classification_report.csv"):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(filename)
    print(f"Classification report saved to {filename}")


def plot_confusion_matrix(y_true, y_pred, classes, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")
