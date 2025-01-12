import numpy as np
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
from datetime import datetime
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier

from sklearn.utils.class_weight import compute_class_weight

import logging

import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.model.CustomGoogleNet import CustomDenseNet
from src.model.CustomResNet import CustomResNet
from src.model.CustomMobileNet import CustomMobileNet


# 1. Preprocessing function
def preprocess_images(images):
    # Resize and normalize images for input into GoogleNet and ResNet-18
    return images / 255.0  # Normalize pixel values to [0, 1]


# 2. Load GoogleNet and ResNet-18 for feature extraction
def load_models(input_shape=(224, 224, 3)):
    # GoogleNet (InceptionV3 is a close alternative)
    googlenet_model = CustomDenseNet()

    # mobilenet_model = CustomMobileNet()

    # ResNet-18 (ResNet50 is used as a substitute)
    resnet_model = CustomResNet()

    return googlenet_model, resnet_model


# 3. Feature Extraction
def extract_features(models, images):
    googlenet_model, resnet_model = models
    # Extract features from GoogleNet
    googlenet_features = googlenet_model.predict(images, verbose=1)

    # Extract features from ResNet
    resnet_features = resnet_model.predict(images, verbose=1)

    # Combine features
    combined_features = np.concatenate([googlenet_features, resnet_features], axis=1)
    return combined_features


# 4. Train and Evaluate Classifiers
def train_and_evaluate(
    X_features, y, classifier_type="SVM", log_dir="logs", model_name="trained_model"
):
    """
    Trains and evaluates a classifier while saving logs and model state after each epoch.

    Parameters:
        X_features (np.ndarray): Extracted features.
        y (np.ndarray): Ground truth labels.
        classifier_type (str): Type of classifier to use (e.g., 'SVM').
        log_dir (str): Directory where logs and model checkpoints will be saved.
        model_name (str): Base name for the saved model files.

    Returns:
        dict: A dictionary of losses for plotting.
    """
    import os
    from datetime import datetime
    import pickle
    import json

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the classifier
    if classifier_type == "SVM":
        model = SVC(kernel="rbf", probability=True)
    elif classifier_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB()
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    # Initialize loss tracking
    losses = {"Training Loss": [], "Validation Loss": []}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    num_of_epochs = 200

    # Simulated training process
    for epoch in range(num_of_epochs):
        # Simulate losses (replace with actual computations)
        train_loss = np.random.rand() * 0.1 + (10 - epoch) * 0.01
        val_loss = np.random.rand() * 0.1 + (10 - epoch) * 0.015

        # Append losses for the epoch
        losses["Training Loss"].append(train_loss)
        losses["Validation Loss"].append(val_loss)

        # Print losses to terminal
        logging(
            f"Epoch {epoch + 1:02d}/{num_of_epochs}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}"
        )

        # Save training logs for this epoch
        log_file = os.path.join(
            log_dir, f"{model_name}_logs_epoch_{epoch + 1}_{timestamp}.json"
        )
        with open(log_file, "w") as f:
            json.dump(losses, f, indent=4)

        # Save model state for this epoch
        model_file = os.path.join(
            log_dir, f"{model_name}_model_epoch_{epoch + 1}_{timestamp}.pkl"
        )
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print(
            f"Checkpoint saved: Epoch {epoch + 1} - Logs: {log_file}, Model: {model_file}"
        )

    # Final model training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print classification report
    print(f"\nClassification Report for {classifier_type}:\n")
    print(classification_report(y_test, y_pred))

    return losses, model


# def extract_features_from_generator(generator, googlenet_model, resnet_model):
#     combined_features = []
#     labels = []

#     for batch_images, batch_labels in generator:
#         # Extract features from GoogleNet
#         googlenet_features = googlenet_model.predict(batch_images, verbose=0)
#         # Extract features from ResNet
#         resnet_features = resnet_model.predict(batch_images, verbose=0)
#         # Combine the features
#         batch_features = np.concatenate([googlenet_features, resnet_features], axis=1)

#         combined_features.append(batch_features)
#         labels.append(batch_labels)

#     # Stack features and labels into arrays
#     combined_features = np.vstack(combined_features)
#     labels = np.concatenate(labels)
#     return combined_features, labels


def extract_features_from_generator(generator, googlenet_model, resnet_model):
    combined_features = []
    labels = []

    print("Starting feature extraction from generator...")

    for i, (batch_images, batch_labels) in enumerate(generator):
        print(f"Processing batch {i + 1}...")  # Debugging: Current batch index
        print(
            f"Batch Images Shape: {batch_images.shape}"
        )  # Debugging: Shape of the image batch
        print(
            f"Batch Labels Shape: {batch_labels.shape}"
        )  # Debugging: Shape of the label batch

        # Extract features from GoogleNet
        print("Extracting features from GoogleNet...")
        googlenet_features = googlenet_model.predict(batch_images, verbose=1)
        print(
            f"GoogleNet Features Shape: {googlenet_features.shape}"
        )  # Debugging: Shape of extracted features

        # Extract features from ResNet
        print("Extracting features from ResNet...")
        resnet_features = resnet_model.predict(batch_images, verbose=1)
        print(
            f"ResNet Features Shape: {resnet_features.shape}"
        )  # Debugging: Shape of extracted features

        # Combine the features
        batch_features = np.concatenate([googlenet_features, resnet_features], axis=1)
        print(
            f"Combined Features Shape: {batch_features.shape}"
        )  # Debugging: Shape after combining features

        # Append features and labels
        combined_features.append(batch_features)
        labels.append(batch_labels)

    # Stack features and labels into arrays
    combined_features = np.vstack(combined_features)
    labels = np.concatenate(labels)

    print("Feature extraction completed.")
    print(
        f"Final Combined Features Shape: {combined_features.shape}"
    )  # Debugging: Final feature shape
    print(f"Final Labels Shape: {labels.shape}")  # Debugging: Final labels shape

    return combined_features, labels


# 5. Train and Evaluate Classifiers Using Generators
def train_and_evaluate_with_generators(
    train_generator,
    validation_generator,
    googlenet_model,
    resnet_model,
    classifier_type="SVM",
    log_dir="logs",
    model_name="trained_model",
    callbacks=None,
):
    if callbacks is None:
        callbacks = []

    print("Started Training.....")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    losses = {"Training Loss": [], "Validation Loss": []}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaler = StandardScaler()

    if classifier_type == "SVM":
        model = SVC(kernel="rbf", probability=True)
    elif classifier_type == "RF":
        model = RandomForestClassifier(n_estimators=100)
    elif classifier_type == "NB":
        model = GaussianNB()
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    num_of_epochs = 1000

    # Trigger callbacks at the start of training
    for callback in callbacks:
        callback.on_train_start()

    try:
        for epoch in range(num_of_epochs):
            # Trigger callbacks at the start of an epoch
            for callback in callbacks:
                if hasattr(callback, "on_epoch_start"):
                    callback.on_epoch_start(epoch)

            X_train, y_train = extract_features_from_generator(
                train_generator, googlenet_model, resnet_model
            )
            X_val, y_val = extract_features_from_generator(
                validation_generator, googlenet_model, resnet_model
            )

            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            train_accuracy = accuracy_score(y_train, y_pred_train)
            val_accuracy = accuracy_score(y_val, y_pred_val)
            losses["Training Loss"].append(1 - train_accuracy)
            losses["Validation Loss"].append(1 - val_accuracy)

            logs = {
                "Training Loss": 1 - train_accuracy,
                "Validation Loss": 1 - val_accuracy,
            }
            print(
                f"Epoch {epoch + 1:02d}/{num_of_epochs}: Training Accuracy = {train_accuracy:.2f}, Validation Accuracy = {val_accuracy:.2f}"
            )

            # Trigger callbacks at the end of an epoch
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

            model_file = os.path.join(
                log_dir, f"{model_name}_model_epoch_{epoch + 1}_{timestamp}.pkl"
            )
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

            log_file = os.path.join(
                log_dir, f"{model_name}_logs_epoch_{epoch + 1}_{timestamp}.json"
            )
            with open(log_file, "w") as f:
                json.dump(
                    {
                        "Train Accuracy": train_accuracy,
                        "Validation Accuracy": val_accuracy,
                    },
                    f,
                )
    except StopIteration:
        print("Training stopped early.")

    # Trigger callbacks at the end of training
    for callback in callbacks:
        callback.on_train_end()

    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    return losses, y_val, y_pred, model


import os
import numpy as np
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import pickle


def extract_features_for_svm(
    image_generator, googlenet_model, resnet_model, scaler=None
):
    """
    Extracts features for the entire dataset and scales them.

    Parameters:
        image_generator: Generator yielding batches of images and labels.
        googlenet_model: Pretrained GoogleNet model for feature extraction.
        resnet_model: Pretrained ResNet model for feature extraction.
        scaler: StandardScaler instance for scaling features. If None, a new scaler is created.

    Returns:
        all_features: Numpy array of combined features for the entire dataset.
        all_labels: Numpy array of class labels for the entire dataset.
        scaler: Fitted StandardScaler instance.
    """
    all_features = []
    all_labels = []

    print("[INFO] Starting feature extraction for all images...")
    batch_count = 0

    for batch_images, batch_labels in image_generator:
        batch_count += 1
        print(f"[INFO] Processing batch {batch_count}...")

        # Track image and label shapes
        print(f"[INFO] Batch {batch_count}: Image shape: {batch_images.shape}")
        print(f"[INFO] Batch {batch_count}: Label shape: {batch_labels.shape}")

        # Extract features
        print(f"[INFO] Extracting GoogleNet features for batch {batch_count}...")
        googlenet_features = googlenet_model.predict(batch_images, verbose=0)
        print(
            f"[INFO] Batch {batch_count}: GoogleNet features shape: {googlenet_features.shape}"
        )

        print(f"[INFO] Extracting ResNet features for batch {batch_count}...")
        resnet_features = resnet_model.predict(batch_images, verbose=0)
        print(
            f"[INFO] Batch {batch_count}: ResNet features shape: {resnet_features.shape}"
        )

        # Combine features
        print(f"[INFO] Combining features for batch {batch_count}...")
        batch_features = np.concatenate([googlenet_features, resnet_features], axis=1)
        print(
            f"[INFO] Batch {batch_count}: Combined features shape: {batch_features.shape}"
        )

        # Convert one-hot encoded labels to class indices
        print(f"[INFO] Converting labels for batch {batch_count}...")
        batch_labels = np.argmax(batch_labels, axis=1)

        # Accumulate features and labels
        print(f"[INFO] Accumulating features and labels for batch {batch_count}...")
        all_features.append(batch_features)
        all_labels.append(batch_labels)
        print(f"[INFO] Batch {batch_count} processed successfully.")

    # Combine all features and labels
    print("[INFO] Combining all batches into a single dataset...")
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    print(
        f"[INFO] Combined dataset shape: Features - {all_features.shape}, Labels - {all_labels.shape}"
    )

    # Scale features
    if scaler is None:
        print("[INFO] Initializing and fitting StandardScaler...")
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)
        print("[INFO] Feature scaling completed with a new StandardScaler.")
    else:
        print("[INFO] Using provided StandardScaler for feature scaling...")
        all_features = scaler.transform(all_features)
        print("[INFO] Feature scaling completed with the provided StandardScaler.")

    print("[INFO] Feature extraction and scaling completed for all images.")
    return all_features, all_labels, scaler


def train_svm_on_full_dataset(
    train_generator,
    validation_generator,
    googlenet_model,
    resnet_model,
    log_dir="logs",
    model_name="trained_svm_model",
):
    """
    Trains an SVM classifier on features extracted from GoogleNet and ResNet models.

    Parameters:
        train_generator: Generator yielding training data.
        validation_generator: Generator yielding validation data.
        googlenet_model: Pretrained GoogleNet model for feature extraction.
        resnet_model: Pretrained ResNet model for feature extraction.
        log_dir: Directory to save logs and the model.
        model_name: Name for the saved model.

    Returns:
        model: Trained SVM model.
        val_labels: True labels for the validation set.
        y_val_pred: Predicted labels for the validation set.
        y_val_prob: Predicted probabilities for the validation set.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("[INFO] Starting Training with Full Dataset for SVM...")

    # Extract features for training data
    print("[INFO] Extracting features for training set...")
    train_features, train_labels, scaler = extract_features_for_svm(
        train_generator, googlenet_model, resnet_model
    )

    # Compute class weights
    print("[INFO] Computing class weights...")
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[INFO] Computed Class Weights: {class_weight_dict}")

    # Initialize SVM with class weights
    print("[INFO] Initializing SVM classifier...")
    model = SVC(kernel="rbf", probability=True, class_weight=class_weight_dict)

    # Train the SVM classifier
    print("[INFO] Training the SVM classifier on the full training dataset...")
    model.fit(train_features, train_labels)

    # Extract features for validation data
    print("[INFO] Extracting features for validation set...")
    val_features, val_labels, _ = extract_features_for_svm(
        validation_generator, googlenet_model, resnet_model, scaler=scaler
    )

    # Evaluate the trained SVM model
    print("[INFO] Evaluating the SVM classifier on validation data...")
    y_val_pred = model.predict(val_features)
    y_val_prob = model.predict_proba(val_features)

    # Validation accuracy
    val_accuracy = accuracy_score(val_labels, y_val_pred)
    print(f"[INFO] Validation Accuracy: {val_accuracy:.4f}")

    # Classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(val_labels, y_val_pred))

    # Log-loss
    val_log_loss = log_loss(val_labels, y_val_prob)
    print(f"[INFO] Validation Log-Loss: {val_log_loss}")

    # Save log-loss to a file
    with open(os.path.join(log_dir, "log_loss.txt"), "w") as f:
        f.write(f"Validation Log-Loss: {val_log_loss}\n")

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(val_labels, y_val_pred)
    plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
    # plt.show()

    # Save the trained model
    model_file = os.path.join(log_dir, f"{model_name}_final_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"[INFO] Trained model saved successfully to {model_file}")
    return model, val_labels, y_val_pred, y_val_prob


import os
import logging
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import pickle

from sklearn.linear_model import SGDClassifier
import numpy as np


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import pickle
import json
import numpy as np
from datetime import datetime
import tensorflow as tf


def build_feature_based_model(input_dim=2000, num_classes=5):
    """
    Builds a neural network for feature-based classification.

    Parameters:
        input_dim (int): Dimension of the input feature vector.
        num_classes (int): Number of output classes.

    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(512, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def incremental_train_classifier_with_epochs(
    train_generator,
    validation_generator,
    googlenet_model,
    resnet_model,
    num_classes,
    classifier_type="NN",
    log_dir="logs",
    model_name="trained_model",
    num_epochs=10,
    callbacks=None,
):
    if callbacks is None:
        callbacks = []

    print("[INFO] Started Incremental Training with Feature Extraction and Epochs...")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    losses = {"Training Loss": [], "Validation Loss": []}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaler = StandardScaler()

    # Precompute class weights
    print("[INFO] Computing class weights...")
    dummy_labels = np.concatenate(
        [np.argmax(labels, axis=1) for _, labels in train_generator]
    )
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_classes), y=dummy_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"[INFO] Computed Class Weights: {class_weights_dict}")

    # Initialize the model
    print(f"[INFO] Initializing classifier: {classifier_type}")
    if classifier_type == "NN":
        model = build_feature_based_model(input_dim=2000, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Trigger callbacks at the start of training
    for callback in callbacks:
        if hasattr(callback, "on_train_start"):
            callback.on_train_start()

    try:
        for epoch in range(1, num_epochs + 1):
            print(f"\n[INFO] Epoch {epoch}/{num_epochs} started.")
            batch_count = 0
            y_train_true = []
            y_train_pred = []

            # Trigger callbacks at the start of an epoch
            for callback in callbacks:
                if hasattr(callback, "on_epoch_start"):
                    callback.on_epoch_start(epoch)

            for batch_images, batch_labels in train_generator:
                batch_count += 1
                print(f"[INFO] Training: Processing batch {batch_count}...")

                # Extract features for the current batch
                googlenet_features = googlenet_model.predict(batch_images, verbose=0)
                resnet_features = resnet_model.predict(batch_images, verbose=0)
                batch_features = np.concatenate(
                    [googlenet_features, resnet_features], axis=1
                )

                # Scale features
                batch_features = scaler.fit_transform(batch_features)

                # If batch_labels are class indices, convert to one-hot
                if len(batch_labels.shape) == 1 or batch_labels.shape[1] != num_classes:
                    batch_labels_one_hot = to_categorical(batch_labels, num_classes)
                else:
                    # Use the labels as-is if they are already one-hot encoded
                    batch_labels_one_hot = batch_labels

                # Train NN incrementally with class weights
                model.fit(
                    batch_features,
                    batch_labels_one_hot,
                    epochs=1,
                    verbose=0,
                    class_weight=class_weights_dict,
                )

                # Track training accuracy for the batch
                y_train_true.extend(np.argmax(batch_labels_one_hot, axis=1))
                y_train_pred.extend(
                    np.argmax(model.predict(batch_features, verbose=0), axis=1)
                )

            # Calculate training accuracy for the epoch
            train_accuracy = accuracy_score(y_train_true, y_train_pred)
            losses["Training Loss"].append(1 - train_accuracy)

            print(f"[INFO] Epoch {epoch} Training Accuracy: {train_accuracy:.4f}")

            # Validation accuracy trackers
            y_val_true = []
            y_val_pred = []

            for batch_images, batch_labels in validation_generator:
                print(f"[INFO] Validation: Processing batch...")

                # Extract features for validation batch
                googlenet_features = googlenet_model.predict(batch_images, verbose=0)
                resnet_features = resnet_model.predict(batch_images, verbose=0)
                batch_features = np.concatenate(
                    [googlenet_features, resnet_features], axis=1
                )

                # Scale features
                batch_features = scaler.transform(batch_features)

                # Predict validation accuracy
                y_pred_batch = np.argmax(
                    model.predict(batch_features, verbose=0), axis=1
                )
                y_val_pred.extend(y_pred_batch)
                y_val_true.extend(np.argmax(batch_labels, axis=1))

            # Calculate validation accuracy for the epoch
            val_accuracy = accuracy_score(y_val_true, y_val_pred)
            losses["Validation Loss"].append(1 - val_accuracy)

            # Trigger callbacks at the end of the epoch
            logs = {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
            for callback in callbacks:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch, logs)

            print(
                f"[INFO] Epoch {epoch}/{num_epochs}: Training Accuracy = {train_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}"
            )

            # Save the model after every epoch
            if classifier_type == "NN":
                # Save model weights
                weights_file = os.path.join(
                    log_dir, f"{model_name}_epoch_{epoch}_{timestamp}.weights.h5"
                )
                model.save_weights(weights_file)

                # Save model architecture as JSON
                architecture_file = os.path.join(
                    log_dir, f"{model_name}_epoch_{epoch}_{timestamp}.json"
                )
                model_json = model.to_json()
                with open(architecture_file, "w") as json_file:
                    json_file.write(model_json)

                print(
                    f"[INFO] Saved NN weights to {weights_file} and architecture to {architecture_file}"
                )

            elif classifier_type == "SGD":
                # Save SGD model using pickle
                model_file = os.path.join(
                    log_dir, f"{model_name}_epoch_{epoch}_{timestamp}.pkl"
                )
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)

                print(f"[INFO] Saved SGD model to {model_file}")

        print("[INFO] Training on all epochs completed.")

        print("\n[INFO] Final Classification Report (Validation):")
        print(classification_report(y_val_true, y_val_pred))

    except StopIteration:
        print("[INFO] Training stopped early.")

    return losses, y_val_true, y_val_pred, model


# Define Focal Loss for both NN and SGD
class FocalLoss:
    def __init__(self, gamma=2.0, alpha=None):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_loss = tf.pow(1 - y_pred, self.gamma) * cross_entropy

        if self.alpha is not None:
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            focal_loss *= alpha_t

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


def incremental_train_classifier_with_epochs_focal_loss(
    train_generator,
    validation_generator,
    googlenet_model,
    resnet_model,
    num_classes,
    classifier_type="NN",
    log_dir="logs",
    model_name="trained_model",
    num_epochs=10,
    callbacks=None,
    gamma=2.0,
    alpha=None,
):
    if callbacks is None:
        callbacks = []

    print("[INFO] Started Incremental Training with Feature Extraction and Epochs...")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    losses = {"Training Loss": [], "Validation Loss": []}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scaler = StandardScaler()

    # Precompute class weights
    print("[INFO] Computing class weights...")
    dummy_labels = np.concatenate(
        [np.argmax(labels, axis=1) for _, labels in train_generator]
    )
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_classes), y=dummy_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"[INFO] Computed Class Weights: {class_weights_dict}")

    # Initialize the model
    print(f"[INFO] Initializing classifier: {classifier_type}")
    if classifier_type == "NN":
        model = build_feature_based_model(input_dim=2000, num_classes=num_classes)
        model.compile(
            optimizer="adam",
            loss=FocalLoss(gamma=gamma, alpha=alpha),
            metrics=["accuracy"],
        )
    elif classifier_type == "SGD":
        model = SGDClassifier(
            loss="log",
            penalty="l2",
            max_iter=1,
            warm_start=True,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    # Trigger callbacks at the start of training
    for callback in callbacks:
        if hasattr(callback, "on_train_start"):
            callback.on_train_start()

    try:
        for epoch in range(1, num_epochs + 1):
            print(f"\n[INFO] Epoch {epoch}/{num_epochs} started.")
            batch_count = 0
            y_train_true = []
            y_train_pred = []

            # Trigger callbacks at the start of an epoch
            for callback in callbacks:
                if hasattr(callback, "on_epoch_start"):
                    callback.on_epoch_start(epoch)

            for batch_images, batch_labels in train_generator:
                batch_count += 1
                print(f"[INFO] Training: Processing batch {batch_count}...")

                # Extract features for the current batch
                googlenet_features = googlenet_model.predict(batch_images, verbose=0)
                resnet_features = resnet_model.predict(batch_images, verbose=0)
                batch_features = np.concatenate(
                    [googlenet_features, resnet_features], axis=1
                )

                # Scale features
                batch_features = scaler.fit_transform(batch_features)

                # If batch_labels are class indices, convert to one-hot
                if len(batch_labels.shape) == 1 or batch_labels.shape[1] != num_classes:
                    batch_labels_one_hot = to_categorical(batch_labels, num_classes)
                else:
                    # Use the labels as-is if they are already one-hot encoded
                    batch_labels_one_hot = batch_labels

                if classifier_type == "NN":
                    # Train NN incrementally
                    model.fit(
                        batch_features,
                        batch_labels_one_hot,
                        epochs=1,
                        verbose=0,
                    )
                elif classifier_type == "SGD":
                    # Train SGD incrementally with class weights
                    sample_weights = np.array(
                        [
                            class_weights_dict[label]
                            for label in np.argmax(batch_labels, axis=1)
                        ]
                    )
                    model.partial_fit(
                        batch_features,
                        np.argmax(batch_labels, axis=1),
                        classes=np.arange(num_classes),
                        sample_weight=sample_weights,
                    )

                # Track training accuracy for the batch
                y_train_true.extend(np.argmax(batch_labels_one_hot, axis=1))
                y_train_pred.extend(
                    np.argmax(model.predict(batch_features, verbose=0), axis=1)
                )

            # Calculate training accuracy for the epoch
            train_accuracy = accuracy_score(y_train_true, y_train_pred)
            losses["Training Loss"].append(1 - train_accuracy)

            print(f"[INFO] Epoch {epoch} Training Accuracy: {train_accuracy:.4f}")

            # Validation accuracy trackers
            y_val_true = []
            y_val_pred = []

            for batch_images, batch_labels in validation_generator:
                print(f"[INFO] Validation: Processing batch...")

                # Extract features for validation batch
                googlenet_features = googlenet_model.predict(batch_images, verbose=0)
                resnet_features = resnet_model.predict(batch_images, verbose=0)
                batch_features = np.concatenate(
                    [googlenet_features, resnet_features], axis=1
                )

                # Scale features
                batch_features = scaler.transform(batch_features)

                # Predict validation accuracy
                y_pred_batch = np.argmax(
                    model.predict(batch_features, verbose=0), axis=1
                )
                y_val_pred.extend(y_pred_batch)
                y_val_true.extend(np.argmax(batch_labels, axis=1))

            # Calculate validation accuracy for the epoch
            val_accuracy = accuracy_score(y_val_true, y_val_pred)
            losses["Validation Loss"].append(1 - val_accuracy)

            # Trigger callbacks at the end of the epoch
            logs = {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}
            for callback in callbacks:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch, logs)

            print(
                f"[INFO] Epoch {epoch}/{num_epochs}: Training Accuracy = {train_accuracy:.4f}, Validation Accuracy = {val_accuracy:.4f}"
            )

            # Save the model after every epoch
            if classifier_type == "NN":
                # Save model weights
                weights_file = os.path.join(
                    log_dir, f"{model_name}_epoch_{epoch}_{timestamp}.weights.h5"
                )
                model.save_weights(weights_file)

                # Save model architecture as JSON
                architecture_file = os.path.join(
                    log_dir, f"{model_name}_epoch_{epoch}_{timestamp}.json"
                )
                model_json = model.to_json()
                with open(architecture_file, "w") as json_file:
                    json_file.write(model_json)

                print(
                    f"[INFO] Saved NN weights to {weights_file} and architecture to {architecture_file}"
                )
            elif classifier_type == "SGD":
                # Save SGD model using pickle
                model_file = os.path.join(
                    log_dir, f"{model_name}_epoch_{epoch}_{timestamp}.pkl"
                )
                with open(model_file, "wb") as f:
                    pickle.dump(model, f)

                print(f"[INFO] Saved SGD model to {model_file}")

        print("[INFO] Training on all epochs completed.")

        print("\n[INFO] Final Classification Report (Validation):")
        print(classification_report(y_val_true, y_val_pred))

    except StopIteration:
        print("[INFO] Training stopped early.")

    return losses, y_val_true, y_val_pred, model


# Main Script
if __name__ == "__main__":
    # Example Dataset (Replace with real data)
    num_samples = 1000
    X = np.random.rand(num_samples, 224, 224, 3)  # Replace with real images
    y = np.random.randint(0, 3, num_samples)

    # Preprocess images
    X_preprocessed = preprocess_images(X)

    # Load GoogleNet and ResNet models
    googlenet_model, resnet_model = load_models()

    # Extract features
    features = extract_features((googlenet_model, resnet_model), X_preprocessed)

    # Train and evaluate SVM
    train_and_evaluate(features, y, classifier_type="svm")
