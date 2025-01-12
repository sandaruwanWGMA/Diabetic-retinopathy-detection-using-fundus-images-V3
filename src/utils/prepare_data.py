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

# def preprocess(
#     kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
#     img_size=(224, 224),
#     batch_size=8,
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

#     # Generator function for creating TensorFlow datasets
#     def flow_from_dataframe(
#         core_idg, in_df, path_col, y_col, shuffle=True, color_mode="rgb"
#     ):
#         ds = tf.data.Dataset.from_tensor_slices(
#             (in_df[path_col].values, np.stack(in_df[y_col].values, axis=0))
#         )
#         if shuffle:
#             ds = ds.shuffle(buffer_size=len(in_df))
#         ds = ds.map(lambda x, y: (core_idg(x), y), num_parallel_calls=tf.data.AUTOTUNE)
#         return ds.batch(batch_size)

#     # Create data loaders
#     train_idg = tf_image_loader(out_size=img_size, vertical_flip=True, color_mode="rgb")
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

#     train_gen = flow_from_dataframe(
#         train_idg, train_df, path_col="path", y_col="level_cat"
#     )
#     valid_gen = flow_from_dataframe(
#         valid_idg, valid_df, path_col="path", y_col="level_cat", shuffle=False
#     )

#     return train_gen, valid_gen


# Usage example:
# train_gen, valid_gen = preprocess()


############ Using SMOTE (Synthetic Minority Oversampling Technique) for Balancing Classes ############


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def preprocess_with_smote(
    kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
    img_size=(224, 224),
    batch_size=8,
):
    """
    Preprocess dataset and apply SMOTE for synthetic oversampling.
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

    import time

    def extract_features(image_paths, img_size):
        """
        Extracts features from images by resizing and flattening them.

        Parameters:
            image_paths (list): List of paths to images.
            img_size (tuple): Target size to resize images (width, height).

        Returns:
            np.array: Array of flattened image features.
        """
        features = []
        start_time = time.time()
        total_images = len(image_paths)

        print(f"[INFO] Starting feature extraction for {total_images} images...")

        for idx, path in enumerate(image_paths):
            # Log progress every 500 images
            if idx % 500 == 0 and idx > 0:
                elapsed_time = time.time() - start_time
                print(
                    f"[INFO] Processed {idx}/{total_images} images "
                    f"(Elapsed time: {elapsed_time:.2f} seconds)"
                )

            # Load, resize, and flatten the image
            img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.image.resize(img, img_size)
            features.append(img.numpy().flatten())  # Flatten the image

        # Calculate total processing time
        total_time = time.time() - start_time
        print(f"[INFO] Feature extraction completed.")
        print(f"[INFO] Total images processed: {total_images}")
        print(f"[INFO] Total time taken: {total_time:.2f} seconds")
        print(f"[INFO] Average time per image: {total_time / total_images:.4f} seconds")

        return np.array(features)

    # Extract features for SMOTE
    print("[INFO] Extracting features for SMOTE...")
    train_features = extract_features(train_df["path"].values, img_size)
    train_labels = train_df["level"].values

    # Apply SMOTE
    print("[INFO] Applying SMOTE to balance training data...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)  # Normalize features
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    smote_features, smote_labels = smote.fit_resample(
        train_features_scaled, train_labels
    )
    print(f"[INFO] SMOTE applied. New training data size: {len(smote_labels)}")

    # Convert SMOTE data back to a DataFrame
    smote_df = pd.DataFrame(smote_features)
    smote_df["level"] = smote_labels
    smote_df["level_cat"] = smote_df["level"].map(
        lambda x: to_categorical(x, 1 + retina_df["level"].max())
    )

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
        train_idg, smote_df, path_col="path", y_col="level_cat"
    )
    valid_gen = flow_from_dataframe(
        valid_idg, valid_df, path_col="path", y_col="level_cat", shuffle=False
    )

    return train_gen, valid_gen


# Usage example:
# train_gen, valid_gen = preprocess_with_smote()
