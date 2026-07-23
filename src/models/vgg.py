"""VGG-style Sequential CNN. See specs.md Step 3."""

from __future__ import annotations

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

from config import IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH


def build_model(*, compile_model: bool = True) -> Sequential:
    model = Sequential(
        [
            Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            ),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, (3, 3), padding="same", activation="relu"),
            Conv2D(256, (3, 3), padding="same", activation="relu"),
            Conv2D(256, (3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), padding="same", activation="relu"),
            Conv2D(512, (3, 3), padding="same", activation="relu"),
            Conv2D(512, (3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), padding="same", activation="relu"),
            Conv2D(512, (3, 3), padding="same", activation="relu"),
            Conv2D(512, (3, 3), padding="same", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation="relu"),
            Dense(4096, activation="relu"),
            Dense(2, activation="softmax"),
        ],
        name="vgg_style_cats_dogs",
    )

    if compile_model:
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def build_transfer_model(*, compile_model: bool = True, trainable_base: bool = False):
    """VGG16 ImageNet backbone + small head — tutorial-aligned and CPU-friendly."""
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
    )
    base.trainable = trainable_base

    x = GlobalAveragePooling2D(name="gap")(base.output)
    x = Dense(256, activation="relu", name="head_dense")(x)
    x = Dropout(0.5, name="head_dropout")(x)
    outputs = Dense(2, activation="softmax", name="predictions")(x)
    model = Model(base.input, outputs, name="vgg16_transfer_cats_dogs")

    if compile_model:
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    import tensorflow as tf

    trainable = int(sum(int(tf.size(w)) for w in model.trainable_weights))
    total = int(model.count_params())
    model._transfer_metadata = {
        "architecture": "vgg16_transfer",
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }
    return model
