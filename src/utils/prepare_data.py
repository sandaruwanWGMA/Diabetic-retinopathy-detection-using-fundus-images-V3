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


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import time


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import time


import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

def preprocess_with_smote(
    kaggle_base_dir="/kaggle/input/diabetic-retinopathy-blindness-detection-c-data",
    img_size=(224, 224),
    batch_size=8,
):
    """
    Preprocess dataset and apply SMOTE for synthetic oversampling with batch-wise processing.
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
    print("[INFO] Training set size:", train_df.shape[0])
    print("[INFO] Validation set size:", valid_df.shape[0])

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

    def process_batch(image_paths, labels, img_size):
        """
        Process a single batch of images and labels.
        """
        batch_features = []
        for path in image_paths:
            img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.image.resize(img, img_size)
            batch_features.append(img.numpy().flatten())
        batch_features = np.array(batch_features)
        return batch_features, labels

    def smote_batch_generator(image_paths, labels, img_size, batch_size, smote):
        """
        Batch generator that applies SMOTE on each batch to avoid memory overflow.
        """
        scaler = StandardScaler()
        start_time = time.time()

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            batch_features, batch_labels = process_batch(batch_paths, batch_labels, img_size)

            # Scale features
            batch_features = scaler.fit_transform(batch_features)

            # Dynamically adjust n_neighbors to avoid errors with small class sizes
            unique_classes, class_counts = np.unique(batch_labels, return_counts=True)
            min_samples = np.min(class_counts)
            n_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            smote.set_params(k_neighbors=n_neighbors)

            # Apply SMOTE to the batch
            smote_features, smote_labels = smote.fit_resample(batch_features, batch_labels)

            elapsed_time = (time.time() - start_time) / 60  # Cumulative elapsed time in minutes
            print(
                f"[INFO] Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images "
                f"(Cumulative elapsed time: {elapsed_time:.2f} minutes)"
            )

            yield smote_features, smote_labels

    # Create the SMOTE batch generator
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    smote_gen = smote_batch_generator(
        train_df["path"].values, train_df["level"].values, img_size, batch_size=500, smote=smote
    )

    # Process batches and create TensorFlow dataset
    def flow_from_smote(smote_gen, num_classes):
        for smote_features, smote_labels in smote_gen:
            smote_labels_cat = to_categorical(smote_labels, num_classes)
            yield smote_features, smote_labels_cat

    def flow_from_dataframe(
        core_idg, in_df, path_col, y_col, shuffle=True, color_mode="rgb"
    ):
        """
        Create a TensorFlow dataset from a Pandas DataFrame.
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (in_df[path_col].values, np.stack(in_df[y_col].values, axis=0))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(in_df))
        ds = ds.map(lambda x, y: (core_idg(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size)

    train_gen = tf.data.Dataset.from_generator(
        lambda: flow_from_smote(smote_gen, num_classes=1 + retina_df["level"].max()),
        output_signature=(
            tf.TensorSpec(shape=(None, img_size[0] * img_size[1] * 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1 + retina_df["level"].max()), dtype=tf.float32),
        ),
    )

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
    valid_gen = flow_from_dataframe(
        valid_idg, valid_df, path_col="path", y_col="level_cat", shuffle=False
    )

    return train_gen.batch(batch_size), valid_gen

# Usage example:
# train_gen, valid_gen = preprocess_with_smote()
