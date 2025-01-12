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
    incremental_train_classifier_with_epochs,
    train_svm_on_full_dataset,
    incremental_train_classifier_with_epochs_focal_loss,
)
from src.utils.plotting import (
    plot_loss,
    save_losses_to_file,
    save_classification_report,
    plot_confusion_matrix,
)

from src.utils.prepare_data import preprocess, preprocess_with_smote
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


train_generator, validation_generator = preprocess_with_smote(batch_size=64)


####### ONLY FOR PRE-PROCESSING WITH SMOTE #######

# Verify generators
try:
    print("[INFO] Verifying data generators...")
    for gen, name in zip([train_generator, validation_generator], ["Training", "Validation"]):
        for _ in gen:
            pass
        print(f"[INFO] {name} generator verified successfully.")
except Exception as e:
    print(f"[ERROR] Data generator verification failed: {e}")
    raise


# Train and evaluate using generators
custom_early_stopping = CustomEarlyStopping(patience=15, min_delta=0.01)

# Load GoogleNet and ResNet models
googlenet_model, resnet_model = load_models()


####### FOR INCREMENTAL TRAINING SGD WITH EPOCHS #######

# num_classes = 5

# # Train classifier incrementally with epochs
# losses, y_val, y_pred, trained_model = incremental_train_classifier_with_epochs(
#     train_generator=train_generator,
#     validation_generator=validation_generator,
#     googlenet_model=googlenet_model,
#     resnet_model=resnet_model,
#     num_classes=num_classes,
#     classifier_type="SGD",
#     log_dir="logs",
#     model_name="diabetic_retinopathy_model",
#     num_epochs=25,
#     callbacks=[custom_early_stopping],
# )

# # Ensure the saved_models directory exists
# saved_models_dir = "saved_models"
# if not os.path.exists(saved_models_dir):
#     os.makedirs(saved_models_dir)

# # Define the path to save the trained model
# model_save_path = os.path.join(saved_models_dir, "trained_model.pkl")

# # Save the model
# with open(model_save_path, "wb") as f:
#     pickle.dump(trained_model, f)

# print(f"Trained model saved successfully to {model_save_path}")


# # Save classification report and plot confusion matrix
# save_classification_report(y_val, y_pred)
# plot_confusion_matrix(y_val, y_pred, classes=[0, 1, 2, 3, 4])

# # Existing plotting and saving functionality
# plot_loss(losses, title="Loss Function Over Time", save_path="loss_plot.png")
# save_losses_to_file(losses, "loss_values.txt")


####### FOR INCREMENTAL TRAINING SGD WITH EPOCHS AND FOCAL LOSS #######

num_classes = 5

# Train classifier incrementally with epochs using focal loss
losses, y_val, y_pred, trained_model = (
    incremental_train_classifier_with_epochs_focal_loss(
        train_generator=train_generator,
        validation_generator=validation_generator,
        googlenet_model=googlenet_model,
        resnet_model=resnet_model,
        num_classes=num_classes,
        classifier_type="SGD",
        log_dir="logs",
        model_name="diabetic_retinopathy_model",
        num_epochs=25,
        callbacks=[custom_early_stopping],
        gamma=2.0,  # Focal Loss hyperparameter
        alpha=0.25,  # Focal Loss hyperparameter for class weighting
    )
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


####### FOR TRAINING THE SVM #######

# # Train the SVM model on the full dataset
# model, val_labels, y_val_pred, y_val_prob = train_svm_on_full_dataset(
#     train_generator=train_generator,  # Training generator
#     validation_generator=validation_generator,  # Validation generator
#     googlenet_model=googlenet_model,  # Pretrained GoogleNet model
#     resnet_model=resnet_model,  # Pretrained ResNet model
#     log_dir="logs",  # Directory to save logs/models
#     model_name="diabetic_retinopathy_model",  # Model name
# )

# # Ensure the saved_models directory exists
# saved_models_dir = "saved_models"
# if not os.path.exists(saved_models_dir):
#     os.makedirs(saved_models_dir)

# # Save the trained model
# model_save_path = os.path.join(saved_models_dir, "trained_model.pkl")
# with open(model_save_path, "wb") as f:
#     pickle.dump(model, f)
# print(f"[INFO] Trained model saved successfully to {model_save_path}")

# # Save classification report using your function
# classification_report_save_path = os.path.join(
#     saved_models_dir, "classification_report.csv"
# )
# save_classification_report(
#     val_labels, y_val_pred, filename=classification_report_save_path
# )

# # Plot and save confusion matrix using your function
# confusion_matrix_save_path = os.path.join(saved_models_dir, "confusion_matrix.png")
# plot_confusion_matrix(
#     val_labels, y_val_pred, classes=[0, 1, 2, 3, 4], filename=confusion_matrix_save_path
# )

####### FOR INCREMENTAL TRAINING WITH EPOCHS USING NEURAL NETWORK #######

# num_classes = 5

# # Train classifier incrementally with epochs using NN
# losses, y_val, y_pred, trained_model = incremental_train_classifier_with_epochs(
#     train_generator=train_generator,
#     validation_generator=validation_generator,
#     googlenet_model=googlenet_model,
#     resnet_model=resnet_model,
#     num_classes=num_classes,
#     classifier_type="NN",
#     log_dir="logs",
#     model_name="diabetic_retinopathy_nn_model",
#     num_epochs=25,
#     callbacks=[custom_early_stopping],
# )

# # Ensure the saved_models directory exists
# saved_models_dir = "saved_models"
# if not os.path.exists(saved_models_dir):
#     os.makedirs(saved_models_dir)

# # Save model weights
# weights_save_path = os.path.join(saved_models_dir, "trained_nn_model.weights.h5")
# trained_model.save_weights(weights_save_path)

# # Save model architecture
# architecture_save_path = os.path.join(saved_models_dir, "trained_nn_model.json")
# model_json = trained_model.to_json()
# with open(architecture_save_path, "w") as json_file:
#     json_file.write(model_json)

# print(f"Trained Neural Network weights saved successfully to {weights_save_path}")
# print(
#     f"Trained Neural Network architecture saved successfully to {architecture_save_path}"
# )

# # Save classification report and plot confusion matrix
# save_classification_report(y_val, y_pred, filename="classification_report_nn.csv")
# plot_confusion_matrix(
#     y_val, y_pred, classes=[0, 1, 2, 3, 4], filename="confusion_matrix_nn.png"
# )

# # Existing plotting and saving functionality
# plot_loss(losses, title="Loss Function Over Time (NN)", save_path="loss_plot_nn.png")
# save_losses_to_file(losses, "loss_values_nn.txt")


####### FOR INCREMENTAL TRAINING WITH EPOCHS USING NEURAL NETWORK AND FOCAL LOSS #######

# num_classes = 5

# # Train classifier incrementally with epochs using NN
# losses, y_val, y_pred, trained_model = (
#     incremental_train_classifier_with_epochs_focal_loss(
#         train_generator=train_generator,
#         validation_generator=validation_generator,
#         googlenet_model=googlenet_model,
#         resnet_model=resnet_model,
#         num_classes=num_classes,
#         classifier_type="NN",
#         log_dir="logs",
#         model_name="diabetic_retinopathy_model",
#         num_epochs=25,
#         callbacks=[custom_early_stopping],
#         gamma=2.0,  # Focal Loss hyperparameter
#         alpha=0.25,  # Focal Loss hyperparameter for class weighting
#     )
# )

# # Ensure the saved_models directory exists
# saved_models_dir = "saved_models"
# if not os.path.exists(saved_models_dir):
#     os.makedirs(saved_models_dir)

# # Save model weights
# weights_save_path = os.path.join(saved_models_dir, "trained_nn_model.weights.h5")
# trained_model.save_weights(weights_save_path)

# # Save model architecture
# architecture_save_path = os.path.join(saved_models_dir, "trained_nn_model.json")
# model_json = trained_model.to_json()
# with open(architecture_save_path, "w") as json_file:
#     json_file.write(model_json)

# print(f"Trained Neural Network weights saved successfully to {weights_save_path}")
# print(
#     f"Trained Neural Network architecture saved successfully to {architecture_save_path}"
# )

# # Save classification report and plot confusion matrix
# save_classification_report(y_val, y_pred, filename="classification_report_nn.csv")
# plot_confusion_matrix(
#     y_val, y_pred, classes=[0, 1, 2, 3, 4], filename="confusion_matrix_nn.png"
# )

# # Existing plotting and saving functionality
# plot_loss(losses, title="Loss Function Over Time (NN)", save_path="loss_plot_nn.png")
# save_losses_to_file(losses, "loss_values_nn.txt")


print("Training complete. Logs, model, reports, and visualizations have been saved.")


####################################

# Computed Class Weights: {0: 0.2770799347471452, 1: 2.6257004830917876, 2: 1.288878349537586, 3: 7.720454545454546, 4: 9.55219683655536}

# Sandruwan WGMA: Epoch 20 SGD with Class weights
# Molindu Sandaruwan: SVM
# Molindu Achintha: Epoch 25 NN

####################################
