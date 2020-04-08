import tensorflow as tf

cutoff_loss = #xxxx decide this

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')>cutoff_loss):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
history = model.fit(x_train, y_train, epochs=50 ...... callbacks=[callbacks])

print(history.history['acc'])
print(history.history['loss'])


#if validation data is provided or validation_split is mentioned

print(history.history['val_acc'])
print(history.history['val_loss'])
