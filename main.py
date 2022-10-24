#1 - import dependencies
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
fdir = '/home/blop/frogeobserver/sorted/frogs/'
tdir = '/home/blop/frogeobserver/sorted/toads/'

#for i in os.listdir(fdir):
#    imgpath = os.path.join(fdir, i)
#    npa = img.imread(imgpath)
#    farr.append(npa)
#for i in os.listdir(tdir):
#    imgpath = os.path.join(tdir, i)
#    npa = img.imread(imgpath)
#    tarr.append(npa)
ftds = tf.keras.utils.image_dataset_from_directory(
    fdir,
    shuffle=True,
    image_size=(224, 224),
    validation_split=0.8,
    subset="training",
    batch_size=None,
    seed=123)
fvds = tf.keras.utils.image_dataset_from_directory(
    fdir,
    shuffle=True,
    image_size=(224, 224),
    validation_split=0.2,
    subset="validation",
    batch_size=None,
    seed=123)
#3 - build and compile model
#4 - fir predict and evaluate