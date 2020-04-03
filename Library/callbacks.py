import json
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf


BATCH_SIZE = *********
X_train = 
y_train = 
x_val = 
y_val =


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

"""We can have an explicit x_val and y_val or we can set validation_split=0.1 
so that 0.1% of the data is used for validation"""

history = model.fit(
    X_train, y_train,
    batch_size = BATCH_SIZE,
    epochs="""Set epochs""",
    callbacks=[kappa_metrics],
    validation_data=(x_val, y_val),
    steps_per_epoch = x_train.shape[0]"""Total number of training examples"""/ BATCH_SIZE
)


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()

plt.plot(kappa_metrics.val_kappas)


"""Do model.predict after this"""