from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.image import resize
import numpy as np


class CustomResNet:
    def __init__(self, num_classes=1000):
        # Initialize the ResNet50 model with the top layer included
        base_model = ResNet50(
            include_top=True,  # Include the fully connected top layer for classification
            weights="imagenet",  # Use pre-trained ImageNet weights
            input_tensor=None,
            input_shape=(224, 224, 3),
            classes=num_classes,  # Specify 1000 classes as per ImageNet training
            classifier_activation="softmax",  # Classifier activation function
            name="resnet50",
        )

        # Freeze all layers in the base model to prevent training
        for layer in base_model.layers:
            layer.trainable = False

        # Create the final model
        self.resnet = Model(inputs=base_model.input, outputs=base_model.output)

    def preprocess_images(self, image_batch):
        """
        Preprocess a batch of 3D image matrices for prediction.

        Args:
            image_batch (np.ndarray): Input images as a 4D batch (batch_size, height, width, channels).

        Returns:
            np.ndarray: Preprocessed image batch ready for prediction.
        """
        # Ensure input is a NumPy array
        if not isinstance(image_batch, np.ndarray):
            raise ValueError(
                "Input must be a NumPy array representing a batch of images."
            )

        # Resize all images in the batch to (224, 224, 3)
        resized_images = np.array(
            [resize(img, (224, 224)).numpy() for img in image_batch]
        )

        # Preprocess the images for ResNet50
        preprocessed_batch = preprocess_input(resized_images)

        return preprocessed_batch

    def summary(self):
        """
        Display the model's architecture.
        """
        return self.resnet.summary()

    def predict(self, image_batch):
        """
        Make predictions using the ResNet50 model.

        Args:
            image_batch (np.ndarray): Input images as a 4D batch (batch_size, height, width, channels).

        Returns:
            np.ndarray: Model predictions (class probabilities).
        """
        # Preprocess the input batch
        preprocessed_batch = self.preprocess_images(image_batch)

        # Make predictions
        predictions = self.resnet.predict(preprocessed_batch)

        return predictions


# # Instantiate the class
# model = CustomResNet()

# # Display model summary
# model.summary()

# # Dummy input batch of images (batch_size=1, height=512, width=512, channels=3)
# dummy_batch = (
#     np.random.rand(5, 512, 512, 3) * 255
# )  # Simulate a batch of RGB images with random values

# # Predict
# predictions = model.predict(dummy_batch)
# print(f"Predicted class probabilities shape:\n{predictions.shape}")
