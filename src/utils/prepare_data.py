# import os
# import numpy as np
# import pandas as pd
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.image import (
#     random_flip_left_right,
#     random_flip_up_down,
#     random_brightness,
#     random_contrast,
#     random_hue,
#     resize,
# )


# def preprocess(
#     kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
#     img_size=(224, 224),
# ):
#     # Load and prepare dataset
#     train_labels_path = os.path.join(kaggle_base_dir, "trainLabels.csv")
#     train_images_dir = os.path.join(kaggle_base_dir, "train_images_768", "train")

#     retina_df = pd.read_csv(train_labels_path)
#     retina_df["PatientId"] = retina_df["image"].map(lambda x: x.split("_")[0])
#     retina_df["path"] = retina_df["image"].map(
#         lambda x: os.path.join(train_images_dir, f"{x}.jpeg")
#     )
#     retina_df["exists"] = retina_df["path"].map(os.path.exists)
#     print(retina_df["exists"].sum(), "images found of", retina_df.shape[0], "total")
#     retina_df["eye"] = retina_df["image"].map(
#         lambda x: 1 if x.split("_")[-1] == "left" else 0
#     )
#     retina_df["level_cat"] = retina_df["level"].map(
#         lambda x: to_categorical(x, 1 + retina_df["level"].max())
#     )

#     retina_df.dropna(inplace=True)
#     retina_df = retina_df[retina_df["exists"]]
#     rr_df = retina_df[["PatientId", "level"]].drop_duplicates()
#     train_ids, valid_ids = train_test_split(
#         rr_df["PatientId"], test_size=0.25, random_state=2018, stratify=rr_df["level"]
#     )
#     train_df = retina_df[retina_df["PatientId"].isin(train_ids)]
#     valid_df = retina_df[retina_df["PatientId"].isin(valid_ids)]
#     print("train", train_df.shape[0], "validation", valid_df.shape[0])

#     # Data augmentation and preprocessing function
#     def tf_image_loader(
#         out_size,
#         horizontal_flip=True,
#         vertical_flip=False,
#         random_brightness=True,
#         random_contrast=True,
#         random_saturation=True,
#         random_hue=True,
#         color_mode="rgb",
#         preproc_func=preprocess_input,
#         on_batch=False,
#     ):
#         def _func(X):
#             X = tf.image.decode_jpeg(
#                 tf.io.read_file(X), channels=3 if color_mode == "rgb" else 0
#             )
#             X = tf.image.resize(X, out_size)  # Resize the image to (224, 224)
#             if horizontal_flip:
#                 X = tf.image.random_flip_left_right(X)
#             if vertical_flip:
#                 X = tf.image.random_flip_up_down(X)
#             if random_brightness:
#                 X = tf.image.random_brightness(X, max_delta=0.1)
#             if random_saturation:
#                 X = tf.image.random_saturation(X, lower=0.75, upper=1.5)
#             if random_hue:
#                 X = tf.image.random_hue(X, max_delta=0.15)
#             if random_contrast:
#                 X = tf.image.random_contrast(X, lower=0.75, upper=1.5)
#             return preproc_func(X)

#         return _func

#     # Generator functions for training and validation
#     def flow_from_dataframe(
#         core_idg, in_df, path_col, y_col, shuffle=True, color_mode="rgb"
#     ):
#         def _func():
#             ds = tf.data.Dataset.from_tensor_slices(
#                 (in_df[path_col].values, np.stack(in_df[y_col].values, 0))
#             )
#             if shuffle:
#                 ds = ds.shuffle(buffer_size=len(in_df))
#             ds = ds.map(
#                 lambda x, y: (core_idg(x), y), num_parallel_calls=tf.data.AUTOTUNE
#             )
#             return ds.batch(batch_size)

#         return _func

#     batch_size = 8
#     core_idg = tf_image_loader(out_size=img_size, vertical_flip=True, color_mode="rgb")
#     valid_idg = tf_image_loader(
#         out_size=img_size,
#         vertical_flip=False,
#         horizontal_flip=False,
#         random_brightness=False,
#         random_contrast=False,
#         random_saturation=False,
#         random_hue=False,
#         color_mode="rgb",
#     )

#     # Create generators
#     train_gen = flow_from_dataframe(
#         core_idg, train_df, path_col="path", y_col="level"
#     )()
#     valid_gen = flow_from_dataframe(
#         valid_idg, valid_df, path_col="path", y_col="level"
#     )()

#     return train_gen, valid_gen


# # Usage example:
# train_gen, valid_gen = preprocess()

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


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time


def preprocess_with_smote(
    kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
    img_size=(224, 224),
    batch_size=8,
):
    """
    Preprocess dataset and apply SMOTE for synthetic oversampling with memory optimization.
    """
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
        preproc_func=tf.keras.applications.inception_v3.preprocess_input,
    ):
        def _func(X):
            X = tf.image.decode_jpeg(
                tf.io.read_file(X), channels=3 if color_mode == "rgb" else 0
            )
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

    def extract_features_in_batches(image_paths, labels, img_size, batch_size=500):
        """
        Extract features in batches to reduce memory usage.
        """
        features = []
        labels_out = []
        scaler = StandardScaler()
        start_time = time.time()

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            batch_features = []
            for path in batch_paths:
                img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
                img = tf.image.resize(img, img_size)
                batch_features.append(img.numpy().flatten())

            batch_features = np.array(batch_features)
            batch_features = scaler.fit_transform(batch_features)
            features.append(batch_features)
            labels_out.append(batch_labels)

            # Log progress
            elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
            print(
                f"[INFO] Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images "
                f"(Elapsed time: {elapsed_time:.2f} minutes)"
            )

        return np.vstack(features), np.concatenate(labels_out)

    # Extract features and apply SMOTE in batches
    print("[INFO] Extracting features and applying SMOTE in batches...")
    train_features, train_labels = extract_features_in_batches(
        train_df["path"].values, train_df["level"].values, img_size, batch_size=500
    )

    smote = SMOTE(sampling_strategy="auto", random_state=42)
    smote_features, smote_labels = smote.fit_resample(train_features, train_labels)

    # Convert SMOTE data back to a DataFrame
    smote_df = pd.DataFrame(smote_features)
    smote_df["level"] = smote_labels
    smote_df["level_cat"] = smote_df["level"].map(
        lambda x: to_categorical(x, 1 + retina_df["level"].max())
    )

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
        train_idg, smote_df, path_col="path", y_col="level_cat"
    )
    valid_gen = flow_from_dataframe(
        valid_idg, valid_df, path_col="path", y_col="level_cat", shuffle=False
    )

    return train_gen, valid_gen


# Usage example:
# train_gen, valid_gen = preprocess_with_smote()
