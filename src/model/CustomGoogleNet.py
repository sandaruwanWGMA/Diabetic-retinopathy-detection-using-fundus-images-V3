from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Resizing
import tensorflow as tf

import numpy as np


class CustomDenseNet:
    def __init__(self, input_tensor=None, input_shape=(512, 512, 3), num_classes=1000):
        # Initialize the EfficientNetB7 model
        base_model = EfficientNetB7(
            include_top=True,  # Include the top layer (fully connected layers)
            weights="imagenet",  # Load pre-trained ImageNet weights
            input_tensor=None,  # Input tensor to use as image input for the model
            input_shape=(
                600,
                600,
                3,
            ),  # Correct input shape for EfficientNetB7 with top
            classes=num_classes,  # Set number of output classes (1000 for ImageNet)
            classifier_activation="softmax",  # Classifier activation function
            name="efficientnetb7",
        )

        # Freeze all layers of the base model (no training)
        base_model.trainable = False

        # If input_tensor is provided, use it as input; otherwise, create a new input layer
        if input_tensor is None:
            input_tensor = tf.keras.layers.Input(shape=input_shape)

        # Resize input images to the required size (600, 600)
        resized_input = Resizing(600, 600, interpolation="bilinear")(input_tensor)

        # Output of the base model
        outputs = base_model(resized_input)

        # Create the model
        self.google_net = Model(inputs=input_tensor, outputs=outputs)

    def summary(self):
        # Display the model's architecture
        return self.google_net.summary()

    def predict(self, images):
        # Make predictions using the frozen base model
        predictions = self.google_net.predict(images)
        return predictions


# # Instantiate the class
# model = CustomDenseNet()

# # Display model summary
# # model.summary()

# # Dummy input image (batch size of 1, 224x224x3 RGB)
# dummy_image = np.random.rand(5, 512, 512, 3) * 255  # Random image

# # Predict
# predictions = model.predict(dummy_image)
# print(f"Shape of predicted class probabilities:\n{predictions.shape}")
