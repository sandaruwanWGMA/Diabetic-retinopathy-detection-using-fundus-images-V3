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
)
from src.utils.plotting import (
    plot_loss,
    save_losses_to_file,
    save_classification_report,
    plot_confusion_matrix,
)

from src.utils.prepare_data import preprocess, generator
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

train_chunks, validation_generator = preprocess(n_splits=10)

# Load models
googlenet_model, resnet_model = load_models()

custom_early_stopping = CustomEarlyStopping(patience=15, min_delta=0.01)

for i, chunk in enumerate(train_chunks):
    print(f"Training on chunk {i + 1}/{len(train_chunks)}")
    train_generator = generator(chunk, path_col="path", y_col="level", batch_size=8)
    
    losses, y_val, y_pred, trained_model = train_and_evaluate_with_generators(
        train_generator=train_generator,
        validation_generator=validation_generator,
        googlenet_model=googlenet_model,
        resnet_model=resnet_model,
        classifier_type="SVM",
        log_dir=f"logs_chunk_{i + 1}",
        model_name=f"custom_trained_model_chunk_{i + 1}",
        callbacks=[custom_early_stopping],
    )
    
    # Save model after training on each chunk
    model_save_path = os.path.join(
        "saved_models", f"trained_model_chunk_{i + 1}.pkl"
    )
    with open(model_save_path, "wb") as f:
        pickle.dump(trained_model, f)

    print(f"Chunk {i + 1} training complete. Model saved to {model_save_path}")

# Save classification report and plot confusion matrix
save_classification_report(y_val, y_pred)
plot_confusion_matrix(y_val, y_pred, classes=[0, 1, 2, 3, 4])

# Existing plotting and saving functionality
plot_loss(losses, title="Loss Function Over Time", save_path="loss_plot.png")
save_losses_to_file(losses, "loss_values.txt")

print("Training complete. Logs, model, reports, and visualizations have been saved.")
