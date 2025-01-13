import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import preprocess_input
from imblearn.over_sampling import SMOTE


############ Using stratified sampling for Balancing Classes ############


def preprocess(
    kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
    img_size=(224, 224),
    batch_size=8,
):
    # Load and prepare dataset
    train_labels_path = os.path.join(kaggle_base_dir, "trainLabels.csv")
    train_images_dir = os.path.join(kaggle_base_dir, "train_images_768", "train")

    retina_df = pd.read_csv(train_labels_path)
    retina_df["PatientId"] = retina_df["image"].map(lambda x: x.split("_")[0])
    retina_df["path"] = retina_df["image"].map(
        lambda x: os.path.join(train_images_dir, f"{x}.jpeg")
    )
    retina_df["exists"] = retina_df["path"].map(os.path.exists)
    print(retina_df["exists"].sum(), "images found of", retina_df.shape[0], "total")
    retina_df["eye"] = retina_df["image"].map(
        lambda x: 1 if x.split("_")[-1] == "left" else 0
    )
    retina_df["level_cat"] = retina_df["level"].map(
        lambda x: to_categorical(x, 1 + retina_df["level"].max())
    )

    retina_df.dropna(inplace=True)
    retina_df = retina_df[retina_df["exists"]]
    rr_df = retina_df[["PatientId", "level"]].drop_duplicates()
    train_ids, valid_ids = train_test_split(
        rr_df["PatientId"], test_size=0.25, random_state=2018, stratify=rr_df["level"]
    )
    train_df = retina_df[retina_df["PatientId"].isin(train_ids)]
    valid_df = retina_df[retina_df["PatientId"].isin(valid_ids)]
    print("train", train_df.shape[0], "validation", valid_df.shape[0])

    # Data augmentation and preprocessing function
    def tf_image_loader(
        out_size,
        horizontal_flip=True,
        vertical_flip=False,
        random_brightness=True,
        random_contrast=True,
        random_saturation=True,
        random_hue=True,
        color_mode="rgb",
        preproc_func=preprocess_input,
    ):
        def _func(X):
            X = tf.image.decode_jpeg(
                tf.io.read_file(X), channels=3 if color_mode == "rgb" else 0
            )
            X = tf.image.resize(X, out_size)  # Resize the image to (224, 224)
            if horizontal_flip:
                X = tf.image.random_flip_left_right(X)
            if vertical_flip:
                X = tf.image.random_flip_up_down(X)
            if random_brightness:
                X = tf.image.random_brightness(X, max_delta=0.1)
            if random_saturation:
                X = tf.image.random_saturation(X, lower=0.75, upper=1.5)
            if random_hue:
                X = tf.image.random_hue(X, max_delta=0.15)
            if random_contrast:
                X = tf.image.random_contrast(X, lower=0.75, upper=1.5)
            return preproc_func(X)

        return _func

    # Generator function for creating TensorFlow datasets
    def flow_from_dataframe(
        core_idg, in_df, path_col, y_col, shuffle=True, color_mode="rgb"
    ):
        ds = tf.data.Dataset.from_tensor_slices(
            (in_df[path_col].values, np.stack(in_df[y_col].values, axis=0))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(in_df))
        ds = ds.map(lambda x, y: (core_idg(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size)

    # Create data loaders
    train_idg = tf_image_loader(out_size=img_size, vertical_flip=True, color_mode="rgb")
    valid_idg = tf_image_loader(
        out_size=img_size,
        vertical_flip=False,
        horizontal_flip=False,
        random_brightness=False,
        random_contrast=False,
        random_saturation=False,
        random_hue=False,
        color_mode="rgb",
    )

    train_gen = flow_from_dataframe(
        train_idg, train_df, path_col="path", y_col="level_cat"
    )
    valid_gen = flow_from_dataframe(
        valid_idg, valid_df, path_col="path", y_col="level_cat", shuffle=False
    )

    return train_gen, valid_gen


# Usage example:
# train_gen, valid_gen = preprocess()


############ Using SMOTE (Synthetic Minority Oversampling Technique) for Balancing Classes ############

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import gc  # Import garbage collector for memory management


def tf_image_loader(
    out_size,
    horizontal_flip=True,
    vertical_flip=False,
    random_brightness=True,
    random_contrast=True,
    random_saturation=True,
    random_hue=True,
    color_mode="rgb",
    preproc_func=tf.keras.applications.inception_v3.preprocess_input,
):
    def _func(X):
        X = tf.io.read_file(X)
        X = tf.image.decode_jpeg(X, channels=3 if color_mode == "rgb" else 0)
        X = tf.image.resize(X, out_size)
        if horizontal_flip:
            X = tf.image.random_flip_left_right(X)
        if vertical_flip:
            X = tf.image.random_flip_up_down(X)
        if random_brightness:
            X = tf.image.random_brightness(X, max_delta=0.1)
        if random_saturation:
            X = tf.image.random_saturation(X, lower=0.75, upper=1.5)
        if random_hue:
            X = tf.image.random_hue(X, max_delta=0.15)
        if random_contrast:
            X = tf.image.random_contrast(X, lower=0.75, upper=1.5)
        return preproc_func(X)

    return _func


def preprocess_with_smote(
    kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
    img_size=(224, 224),
    batch_size=8,
    limit_data=None,
):
    train_labels_path = os.path.join(kaggle_base_dir, "trainLabels.csv")

    # Load the dataset, limiting entries if limit_data is specified
    if limit_data is not None:
        print(f"[INFO] Loading dataset limited to {limit_data} entries.")
        retina_df = pd.read_csv(train_labels_path).head(limit_data)
    else:
        print("[INFO] Loading the full dataset.")
        retina_df = pd.read_csv(train_labels_path)

    train_images_dir = os.path.join(kaggle_base_dir, "train_images_768", "train")
    retina_df["path"] = retina_df["image"].apply(
        lambda x: os.path.join(train_images_dir, f"{x}.jpeg")
    )
    retina_df["exists"] = retina_df["path"].apply(os.path.exists)
    retina_df = retina_df[retina_df["exists"]]
    retina_df["level_cat"] = retina_df["level"].apply(
        lambda x: to_categorical(x, num_classes=5)
    )

    train_df, valid_df = train_test_split(
        retina_df, test_size=0.25, random_state=42, stratify=retina_df["level"]
    )
    print("[INFO] Dataset split into training and validation.")

    image_augmentation = tf_image_loader(img_size)

    def smote_generator(df):
        scaler = StandardScaler()
        smote = SMOTE(random_state=42)
        print("[INFO] Starting SMOTE processing and data augmentation.")

        paths = df["path"].tolist()
        labels = np.stack(df["level_cat"].tolist())
        features = np.array([image_augmentation(path) for path in paths])

        features = features.reshape(len(features), -1)
        features = scaler.fit_transform(features)
        features, labels = smote.fit_resample(features, labels)

        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]

        features = features.reshape(-1, *img_size, 3)
        start_time = time.time()  # Initialize timer
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            elapsed_time = time.time() - start_time
            print(
                f"[INFO] Processing batch {start//batch_size + 1}: Elapsed time: {elapsed_time:.2f} seconds"
            )
            yield features[start:end], labels[start:end]
            gc.collect()

    train_dataset = tf.data.Dataset.from_generator(
        lambda: smote_generator(train_df),
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
        ),
    )
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.array([image_augmentation(path) for path in valid_df["path"].tolist()]),
            np.stack(valid_df["level_cat"].tolist()),
        )
    ).batch(batch_size)

    # Extract one sample to check shapes
    sample_train_features, sample_train_labels = next(iter(train_dataset))
    sample_valid_features, sample_valid_labels = next(iter(valid_dataset))

    print(
        f"[INFO] Example training data shape: {sample_train_features.shape}, {sample_train_labels.shape}"
    )
    print(
        f"[INFO] Example validation data shape: {sample_valid_features.shape}, {sample_valid_labels.shape}"
    )

    # Calculate and print final class counts
    final_train_class_counts = calculate_class_counts(
        train_dataset, num_batches=int(len(train_df) / batch_size)
    )
    final_valid_class_counts = calculate_class_counts(
        valid_dataset, num_batches=int(len(valid_df) / batch_size)
    )
    print("[INFO] Final training class counts:")
    for i, count in enumerate(final_train_class_counts):
        print(f"Class {i+1:02}: {int(count)}")
    print("[INFO] Final validation class counts:")
    for i, count in enumerate(final_valid_class_counts):
        print(f"Class {i+1:02}: {int(count)}")

    return train_dataset, valid_dataset


def calculate_class_counts(generator, num_batches=100):
    class_counts = np.zeros(5)
    for i, (features, labels) in enumerate(generator.take(num_batches)):
        class_counts += np.sum(labels, axis=0)
    return class_counts
