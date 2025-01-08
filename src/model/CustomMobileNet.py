from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Resizing
import tensorflow as tf
import numpy as np


class CustomMobileNet:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000):
        input_tensor = Input(shape=input_shape)

        base_model = MobileNet(
            include_top=True,
            weights="imagenet",
            input_tensor=input_tensor,
            classes=num_classes,
        )

        base_model.trainable = False
        outputs = base_model.output
        self.google_net = Model(inputs=input_tensor, outputs=outputs)

    def summary(self):
        return self.google_net.summary()

    def predict(self, images, **kwargs):
        predictions = self.google_net.predict(images, **kwargs)
        return predictions


# # Instantiate the class
# model = CustomMobileNet()

# # Display model summary
# model.summary()

# # Example usage:
# # Ensure to preprocess your images accordingly if needed before prediction
# dummy_batch = np.random.rand(8, 224, 224, 3) * 255  # Example batch of images
# predictions = model.predict(dummy_batch)
# print(f"Shape of predicted class probabilities: {predictions.shape}")
