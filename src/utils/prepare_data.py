import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.image import (
    random_flip_left_right,
    random_flip_up_down,
    random_brightness,
    random_contrast,
    random_hue,
    resize,
)


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


# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from keras.applications.inception_v3 import preprocess_input
# from keras.preprocessing.image import load_img, img_to_array

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
#     def preprocess_image(image_path):
#         img = load_img(image_path, target_size=img_size)  # Load and resize image
#         img = img_to_array(img)  # Convert to array
#         img = preprocess_input(img)  # Apply preprocessing (InceptionV3)
#         return img

#     # Generator function
#     def generator(df, path_col, y_col, batch_size):
#         while True:
#             df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
#             for start in range(0, len(df), batch_size):
#                 end = min(start + batch_size, len(df))
#                 batch_df = df[start:end]
#                 images = np.array(
#                     [preprocess_image(x) for x in batch_df[path_col].values]
#                 )
#                 labels = np.stack(batch_df[y_col].values, axis=0)
#                 yield images, labels

#     # Create generators
#     train_gen = generator(train_df, path_col="path", y_col="level", batch_size=batch_size)
#     valid_gen = generator(valid_df, path_col="path", y_col="level", batch_size=batch_size)

#     return train_gen, valid_gen


# # Usage example:
# train_gen, valid_gen = preprocess()


# Split dataframe into chunks
def split_dataframe(df, n_splits):
    """Split a dataframe into n_splits parts."""
    return np.array_split(df, n_splits)


# Image augmentation function
def augment_image(image):
    """Apply data augmentation to an image."""
    image = random_flip_left_right(image)
    image = random_flip_up_down(image)
    image = random_brightness(image, max_delta=0.1)
    image = random_contrast(image, lower=0.75, upper=1.5)
    image = random_hue(image, max_delta=0.1)
    return image


# Generator function
def generator(df, path_col, y_col, batch_size, img_size=(224, 224), augment=False):
    """Generator function for creating batches of data."""
    while True:
        df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_df = df[start:end]
            images = []
            for img_path in batch_df[path_col].values:
                # Load and preprocess the image
                img = load_img(img_path, target_size=img_size)
                img = img_to_array(img)
                if augment:
                    img = augment_image(img)  # Apply augmentation
                images.append(img)
            images = np.array(images)
            labels = np.stack(batch_df[y_col].values, axis=0)
            yield images, labels


# Preprocess function
def preprocess(
    kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
    img_size=(224, 224),
    batch_size=8,
    n_splits=10,
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

    # Split training data into chunks
    train_chunks = split_dataframe(train_df, n_splits)

    # Create validation generator (no augmentation)
    valid_gen = generator(
        valid_df, path_col="path", y_col="level", batch_size=batch_size, augment=False
    )

    return train_chunks, valid_gen
