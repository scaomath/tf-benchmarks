#%% quick tf benchmark using Keras with iPython mode on copied from Kaggle
import tensorflow as tf
import numpy as np               
import pandas as pd                 
import matplotlib.pyplot as plt
import keras as k
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from keras import backend as K
import time

import sys
sys.path.append('/home/scao/anaconda3/lib/python3.8/site-packages')
import seaborn as sns
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# %% load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = 28,28
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_test=x_test.astype('float32')
x_train=x_train.astype('float32')
mean=np.mean(x_train)
std=np.std(x_train)
x_test = (x_test-mean)/std
x_train = (x_train-mean)/std
num_classes=10
y_train = k.utils.to_categorical(y_train, num_classes)
y_test = k.utils.to_categorical(y_test, num_classes)


#%% add a time callbacks
class TimePerEpoch(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# %% simple cnn
num_filter=32
num_dense=512
drop_dense=0.7
ac='relu'
lr=1e-3

#%% GPU
with tf.device("/GPU:0"):
    model = Sequential()

    model.add(Conv2D(num_filter, (3, 3), activation=ac, input_shape=(28, 28, 1),padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(num_filter, (3, 3), activation=ac,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 14x14x32

    model.add(Conv2D(2*num_filter, (3, 3), activation=ac,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(2*num_filter, (3, 3), activation=ac,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 7x7x64 = 3136 neurons

    model.add(Flatten())                        
    model.add(Dense(num_dense, activation=ac))
    model.add(BatchNormalization())
    model.add(Dropout(drop_dense))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                metrics=['accuracy'],
                optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))

    

    times_gpu = []
    for i in range(6):
        k=16*2**i
        print(f"\nBatch size {str(k)}")
        time_callback = TimePerEpoch()
        model.fit(x_train, y_train, batch_size=k, epochs=1, 
                    callbacks=[time_callback],
                     validation_data=(x_test, y_test))
        times_gpu.append(time_callback.times)

# %% CPU
with tf.device("/CPU:0"):
    model = Sequential()

    model.add(Conv2D(num_filter, (3, 3), activation=ac, input_shape=(28, 28, 1),padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(num_filter, (3, 3), activation=ac,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 14x14x32

    model.add(Conv2D(2*num_filter, (3, 3), activation=ac,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(2*num_filter, (3, 3), activation=ac,padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # reduces to 7x7x64 = 3136 neurons

    model.add(Flatten())                        
    model.add(Dense(num_dense, activation=ac))
    model.add(BatchNormalization())
    model.add(Dropout(drop_dense))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                metrics=['accuracy'],
                optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    times_cpu = []
    for i in range(6):
        k=16*2**i
        print(f"\nBatch size {str(k)}")
        time_callback = TimePerEpoch()
        model.fit(x_train, y_train, batch_size=k, epochs=1, 
                    callbacks=[time_callback],
                     validation_data=(x_test, y_test))
        times_cpu.append(time_callback.times)


# %% comparison
plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})

sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

batch_sizes=[16*2**i for i in range(6)]
# df = pd.DataFrame(np.c_[batch_sizes, times_cpu, times_gpu], columns=["batch_size", "cpu time", "gpu time"])

plt.plot(batch_sizes,times_gpu,'b--o')
plt.plot(batch_sizes,times_cpu,'r--o')

plt.ylabel('Training time per epoch (seconds)')
plt.xlabel('Batch size')
plt.legend(['gpu', 'cpu'], loc='upper right')
plt.ylim([0,100])
plt.savefig('./10850k_vs_3090.png') 
plt.show()