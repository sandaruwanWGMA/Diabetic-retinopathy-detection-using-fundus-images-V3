from tensorflow.keras.callbacks import Callback, EarlyStopping


# Custom EarlyStopping callback
class CustomEarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float("inf")

    def on_train_start(self):
        print("Training has started. Monitoring for early stopping conditions.")

    def on_epoch_start(self, epoch):
        print(f"Epoch {epoch + 1} has started.")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("Validation Loss", float("inf"))
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                raise StopIteration
