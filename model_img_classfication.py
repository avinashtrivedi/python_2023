import json
import numpy as np
import tensorflow as tf
import os
import shutil
import PIL
import PIL.Image
import matplotlib.pyplot as plt
from tensorflow import keras
import pathlib
import glob
import matplotlib.image as mpimg
#import splitfolders
#import cv2
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

data = tf.keras.utils.image_dataset_from_directory('images')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next() 




#scale data
data = data.map(lambda x, y: (x/255,y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()
print(batch)
print(batch[0].min())
print(batch[0].max())


#plot images
fig1, ax = plt.subplots(ncols=8, figsize=(20,20))
for idx, img in enumerate(batch[0][:8]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])
plt.show()

print(len(data))
# split data

train_size = int(len(data)*.15)
val_size = int(len(data)*.05)
test_size = int(len(data)*.8)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
print(len(train))
print(len(val))
print(len(test))


model = Sequential()
model.add(Conv2D(16,(5,5), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

#train
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train,epochs=50,validation_data=val,callbacks=[tensorboard_callback])

#plot performence
import matplotlib.pyplot as plt
fig2 = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig2.suptitle('loss',fontsize=20)
plt.legend(loc="upper left")
plt.show()


fig3 = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig3.suptitle('Accuracy',fontsize=20)
plt.legend(loc="upper left")
plt.show()

#evaliuate performance
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import confusion_matrix
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
y_pred = []
y_actual =  []

for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y,yhat)
    re.update_state(y,yhat)
    acc.update_state(y,yhat)
    y_actual.extend(y)
    y_pred.extend([1 if i > 0.5 else 0 for i in yhat])
print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')
print('confusion matrix')
print(confusion_matrix(y_actual, y_pred)) 

# save the model

from tensorflow.keras.models import load_model

model.save(os.path.join('models','OneLModel.h5'))
new_model = load_model(os.path.join('models','OneLModel.h5'))
