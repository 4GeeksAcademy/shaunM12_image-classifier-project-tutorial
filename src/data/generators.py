"""Keras ImageDataGenerator helpers for train/val/test. See specs.md Step 2."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from config import (
    BATCH_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    PROCESSED_TEST_DIR,
    PROCESSED_TRAIN_DIR,
    PROCESSED_VAL_DIR,
)


@dataclass
class GeneratorBundle:
    train: DirectoryIterator
    val: DirectoryIterator
    test: DirectoryIterator
    steps_per_epoch: int
    validation_steps: int
    test_steps: int


def _make_datagen(*, augment: bool, use_vgg16_preprocess: bool) -> ImageDataGenerator:
    if use_vgg16_preprocess:
        from tensorflow.keras.applications.vgg16 import preprocess_input

        if augment:
            return ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=20,
                horizontal_flip=True,
            )
        return ImageDataGenerator(preprocessing_function=preprocess_input)

    if augment:
        return ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            horizontal_flip=True,
        )
    return ImageDataGenerator(rescale=1.0 / 255)


def build_generators(
    batch_size: int | None = None,
    *,
    use_vgg16_preprocess: bool = False,
) -> GeneratorBundle:
    batch_size = batch_size or BATCH_SIZE

    train_datagen = _make_datagen(augment=True, use_vgg16_preprocess=use_vgg16_preprocess)
    val_test_datagen = _make_datagen(augment=False, use_vgg16_preprocess=use_vgg16_preprocess)

    train = train_datagen.flow_from_directory(
        str(PROCESSED_TRAIN_DIR),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val = val_test_datagen.flow_from_directory(
        str(PROCESSED_VAL_DIR),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    test = val_test_datagen.flow_from_directory(
        str(PROCESSED_TEST_DIR),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return GeneratorBundle(
        train=train,
        val=val,
        test=test,
        steps_per_epoch=math.ceil(train.samples / batch_size),
        validation_steps=math.ceil(val.samples / batch_size),
        test_steps=math.ceil(test.samples / batch_size),
    )
