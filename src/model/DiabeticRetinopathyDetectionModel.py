import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from CustomGoogleNet import CustomDenseNet
from CustomResNet import CustomResNet

import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class DiabeticRetinopathyDetectionModel:
    def __init__(self):
        # Load pre-trained DenseNet and ResNet models
        self.densenet = CustomDenseNet().get_model()
        self.resnet = CustomResNet().get_model()

        # Initialize SVM model with RBF kernel
        self.svm_model = svm.SVC(kernel="rbf", probability=True)
        self.calibrated_svm = CalibratedClassifierCV(self.svm_model, method="sigmoid")

        # Scaler for feature normalization
        self.scaler = StandardScaler()

    def extract_features(self, input_data):
        # Extract features from DenseNet and ResNet models
        densenet_output = self.densenet.predict(input_data)
        resnet_output = self.resnet.predict(input_data)

        # Concatenate features from both models
        combined_output = np.concatenate([densenet_output, resnet_output], axis=1)
        return combined_output

    def train(self, X_train, y_train):
        # Extract features from training data
        X_features = self.extract_features(X_train)

        # Normalize the features
        X_features = self.scaler.fit_transform(X_features)

        # Train the calibrated SVM model on the extracted features
        self.calibrated_svm.fit(X_features, y_train)
        print("SVM model trained successfully.")

    def predict(self, X_test):
        # Extract and normalize features from test data
        X_features = self.extract_features(X_test)
        X_features = self.scaler.transform(X_features)

        # Predict probabilities for each class
        y_prob = self.calibrated_svm.predict_proba(X_features)
        return y_prob

    def evaluate(self, X_test, y_test):
        # Generate predictions
        y_pred = self.calibrated_svm.predict(
            self.scaler.transform(self.extract_features(X_test))
        )

        # Print classification report
        print(classification_report(y_test, y_pred))


# Example usage
# Load or create data (dummy example data here)
num_samples = 100
X = np.random.rand(num_samples, 224, 224, 3)  # Assume input images are 224x224 RGB
y = np.random.randint(0, 5, num_samples)  # Assuming 5 classes for the task

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = DiabeticRetinopathyDetectionModel()
model.train(X_train, y_train)

# Predict on test set
y_prob = model.predict(X_test)
print("Predicted probabilities for the test set:", y_prob)

# Evaluate the model
model.evaluate(X_test, y_test)
