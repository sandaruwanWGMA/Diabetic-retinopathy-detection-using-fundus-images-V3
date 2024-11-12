import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


class CustomDenseNet:
    def __init__(self):
        # Load the pre-trained DenseNet121 model with ImageNet weights
        self.base_model = DenseNet121(weights="imagenet")

        # Freeze all layers except the last one
        for layer in self.base_model.layers[:-1]:
            layer.trainable = False

    def get_model(self):
        return self.base_model


# Instantiate the custom DenseNet model
# custom_densenet = CustomDenseNet().get_model()
# custom_densenet.summary()

# random_matrix = tf.random.uniform(shape=(1, 224, 224, 3), minval=0, maxval=1)
# prediction = custom_densenet.predict(random_matrix)

# print(prediction.shape)
