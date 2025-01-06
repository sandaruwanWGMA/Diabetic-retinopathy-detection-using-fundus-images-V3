from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import numpy as np


class CustomResNet:
    def __init__(self, num_classes=1000):
        # Initialize the ResNet50 model with the top layer included
        base_model = ResNet50(
            include_top=True,  # Include the fully connected top layer for classification
            weights="imagenet",  # Use pre-trained ImageNet weights
            input_tensor=None,
            input_shape=(
                224,
                224,
                3,
            ),
            classes=num_classes,  # Specifying 1000 classes as per the original ImageNet training
            classifier_activation="softmax",  # Classifier activation function
            name="resnet50",
        )

        # Freeze all layers in the base model to prevent training
        for layer in base_model.layers:
            layer.trainable = False

        # Create the final model
        self.resnet = Model(inputs=base_model.input, outputs=base_model.output)

    def summary(self):
        # Display the model's architecture
        return self.resnet.summary()

    def predict(self, images):
        # Make predictions using the frozen ResNet50 model
        predictions = self.resnet.predict(images)
        return predictions


# # Instantiate the class
# model = CustomResNet()

# # Display model summary
# model.summary()

# # Dummy input image (batch size of 1, 224x224x3 RGB)
# dummy_image = np.random.rand(5, 224, 224, 3) * 255

# # Predict
# predictions = model.predict(dummy_image)
# print(f"Predicted class probabilities:\n{predictions.shape}")
