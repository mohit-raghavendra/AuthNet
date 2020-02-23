import json
import math
import os

import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet121
from keras.applications import inception_v3
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

from face_recog_load_data import *

print("Called Loaddata")
data_df = load_data()

print("Called Preprocessing")
x_train, x_val, y_train, y_val = preprocess_data(data_df)



model = build_model_Inception()
model.summary()

BATCH_SIZE = 32

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2020)


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print("val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return 

kappa_metrics = Metrics()

history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[kappa_metrics]
)

with open('history.json', 'w') as f:
    json.dump(history.history, f)


y_pred = model.predict(x_val) 
print("0")
print(y_pred)


y_val_1d = np.ndarray(y_val.shape[0])
for i in range(y_val.shape[0]):
    y_val_1d[i] = y_val[i].sum()


y_pred_1d = np.ndarray(y_pred.shape[0])
for i in range(y_pred.shape[0]):
    y_pred_1d[i] = y_pred[i].sum()

print(confusion_matrix(y_val_1d, y_pred_1d))

