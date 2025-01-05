import json
import os
from datetime import datetime
import pickle


def save_training_logs_and_model(
    logs, model, model_name="trained_model", log_dir="logs"
):
    """
    Saves training logs and model state to disk.

    Parameters:
        logs (dict): Training logs (e.g., losses, metrics, etc.).
        model (object): Trained model instance.
        model_name (str): Name for saving the model.
        log_dir (str): Directory where logs and model will be saved.
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save training logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_logs_{timestamp}.json")
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)
    print(f"Training logs saved to: {log_file}")

    # Save model state (SVM, RandomForest, etc., using pickle)
    model_file = os.path.join(log_dir, f"{model_name}_model_{timestamp}.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Model state saved to: {model_file}")
