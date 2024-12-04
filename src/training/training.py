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
