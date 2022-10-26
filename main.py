#1 - import dependencies
#%%
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#2 - load data
#%%
tr_dir = '/home/blop/frogeobserver/dataset/train'
va_dir = '/home/blop/frogeobserver/dataset/validation'

tds = keras.preprocessing.image_dataset_from_directory(
    tr_dir,
    shuffle=True,
    image_size=(224, 224),
    batch_size=5,
    seed=123)
vds = tf.keras.utils.image_dataset_from_directory(
    va_dir,
    shuffle=True,
    image_size=(224, 224),
    batch_size=5,
    seed=123)
class_names = tds.class_names
print(class_names)

#3 - build and compile model
#%%
model = Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True),
    metrics=['accuracy'])
model.summary()
#4 - fit predict and evaluate
#%%
model.fit(
    x=tds,
    validation_data=vds,
    epochs=10,
    batch_size=5
    )