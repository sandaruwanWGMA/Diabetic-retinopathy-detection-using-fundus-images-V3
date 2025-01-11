import numpy as np

import os
import sys
import pickle

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=0"


project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.model.DiabeticRetinopathyDetectionModel import (
    load_models,
    train_and_evaluate_with_generators,
    extract_features_from_generator,
    train_classifier_with_extracted_features,
    incremental_train_classifier_with_epochs,
)
from src.utils.plotting import (
    plot_loss,
    save_losses_to_file,
    save_classification_report,
    plot_confusion_matrix,
)

from src.utils.prepare_data import preprocess
from src.utils.schedulers import CustomEarlyStopping


# class DummyGenerator:
#     def __init__(self, num_samples, batch_size, target_size, num_classes):
#         self.num_samples = num_samples
#         self.samples = num_samples
#         self.batch_size = batch_size
#         self.target_size = target_size
#         self.num_classes = num_classes

#     def __len__(self):
#         return int(np.ceil(self.num_samples / self.batch_size))

#     def __iter__(self):
#         for _ in range(self.__len__()):
#             # Generate dummy image data
#             batch_images = np.random.rand(self.batch_size, *self.target_size, 3)
#             # Generate dummy class indices as 1D array
#             batch_labels = np.random.randint(0, self.num_classes, self.batch_size)
#             yield batch_images, batch_labels


# train_generator = DummyGenerator(
#     num_samples=1000, batch_size=32, target_size=(224, 224), num_classes=3
# )
# validation_generator = DummyGenerator(
#     num_samples=200, batch_size=32, target_size=(224, 224), num_classes=3
# )

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"GPUs detected: {[gpu.name for gpu in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # Avoid memory issues
else:
    print("No GPU detected. Training will fall back to CPU.")


train_generator, validation_generator = preprocess()

# Train and evaluate using generators
custom_early_stopping = CustomEarlyStopping(patience=15, min_delta=0.01)

# Load GoogleNet and ResNet models
googlenet_model, resnet_model = load_models()

# Train classifier incrementally with epochs
losses, y_val, y_pred, trained_model = incremental_train_classifier_with_epochs(
    train_generator=train_generator,
    validation_generator=validation_generator,
    googlenet_model=googlenet_model,
    resnet_model=resnet_model,
    classifier_type="SGD",
    log_dir="logs",
    model_name="diabetic_retinopathy_model",
    num_epochs=100,
    callbacks=[custom_early_stopping],
)

# Ensure the saved_models directory exists
saved_models_dir = "saved_models"
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

# Define the path to save the trained model
model_save_path = os.path.join(saved_models_dir, "trained_model.pkl")

# Save the model
with open(model_save_path, "wb") as f:
    pickle.dump(trained_model, f)

print(f"Trained model saved successfully to {model_save_path}")


# Save classification report and plot confusion matrix
save_classification_report(y_val, y_pred)
plot_confusion_matrix(y_val, y_pred, classes=[0, 1, 2, 3, 4])

# Existing plotting and saving functionality
plot_loss(losses, title="Loss Function Over Time", save_path="loss_plot.png")
save_losses_to_file(losses, "loss_values.txt")

print("Training complete. Logs, model, reports, and visualizations have been saved.")
