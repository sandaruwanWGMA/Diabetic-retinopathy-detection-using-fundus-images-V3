import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np


class CustomResNet:
    def __init__(self):
        # Load the pre-trained DenseNet121 model with ImageNet weights
        self.base_model = ResNet50(weights="imagenet")

        # Freeze all layers except the last one
        for layer in self.base_model.layers[:-1]:
            layer.trainable = False

    def get_model(self):
        return self.base_model


# Instantiate the custom DenseNet model
# custom_resnet = CustomResNet().get_model()
# custom_densenet.summary()

# random_matrix = tf.random.uniform(shape=(1, 224, 224, 3), minval=0, maxval=1)
# prediction = custom_resnet.predict(random_matrix)

# print(prediction.shape)
