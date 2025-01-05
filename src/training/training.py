import numpy as np

import os
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.model.DiabeticRetinopathyDetectionModel import (
    load_models,
    train_and_evaluate_with_generators,
)
from src.utils.plotting import (
    plot_loss,
    save_losses_to_file,
    save_classification_report,
    plot_confusion_matrix,
)

from src.utils.prepare_data import preprocess


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

# Load GoogleNet and ResNet models

train_generator, validation_generator = preprocess()

googlenet_model, resnet_model = load_models()

# Train and evaluate using generators
losses, trained_model = train_and_evaluate_with_generators(
    train_generator,
    validation_generator,
    googlenet_model,
    resnet_model,
    classifier_type="SVM",
    log_dir="logs",
    model_name="SVM_DiabeticRetinopathy",
)

losses, y_test, y_pred, trained_model = train_and_evaluate_with_generators(
    train_generator,
    validation_generator,
    googlenet_model,
    resnet_model,
    classifier_type="SVM",
    log_dir="logs",
    model_name="SVM_DiabeticRetinopathy",
)

# Save classification report and plot confusion matrix
save_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, classes=[0, 1, 2, 3, 4])

# Existing plotting and saving functionality
plot_loss(losses, title="Loss Function Over Time", save_path="loss_plot.png")
save_losses_to_file(losses, "loss_values.txt")

print("Training complete. Logs, model, reports, and visualizations have been saved.")
