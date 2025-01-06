from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Resizing
import tensorflow as tf


class CustomDenseNet:
    def __init__(self, input_shape=(512, 512, 3), num_classes=1000):
        # Input layer with the initial shape
        input_tensor = Input(shape=input_shape)

        # Resize input images to the required size (224, 224)
        resized_input = Resizing(224, 224, interpolation="bilinear")(input_tensor)

        # Initialize the EfficientNetB7 model with the resized inputs
        base_model = EfficientNetB7(
            include_top=True,  # Include the top layer (fully connected layers)
            weights="imagenet",  # Load pre-trained ImageNet weights
            input_tensor=resized_input,  # Use the resized input as the input tensor for the model
            classes=num_classes,  # Set number of output classes (1000 for ImageNet)
            classifier_activation="softmax",  # Classifier activation function
        )

        # Freeze all layers of the base model (no training)
        base_model.trainable = False

        # Output of the base model
        outputs = base_model.output

        # Create the model
        self.google_net = Model(inputs=input_tensor, outputs=outputs)

    def summary(self):
        # Display the model's architecture
        return self.google_net.summary()

    def predict(self, images):
        # Make predictions using the frozen base model
        predictions = self.google_net.predict(images)
        return predictions


# Instantiate the class
model = CustomDenseNet()

# Display model summary
model.summary()

# Example usage:
# Ensure to preprocess your images accordingly if needed before prediction
# dummy_batch = np.random.rand(8, 512, 512, 3)  # Example batch of images
# predictions = model.predict(dummy_batch)
# print(f"Shape of predicted class probabilities: {predictions.shape}")
