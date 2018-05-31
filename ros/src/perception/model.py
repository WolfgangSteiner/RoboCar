from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.callbacks import  ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from sklearn.model_selection import train_test_split
from Common import load_data
from DataGenerator import DataGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dirs', metavar='dir', type=str, nargs='+', help="Directory containing training data.")
parser.add_argument('--nval', type=int, default=1024)
parser.add_argument('--mval', type=int, default=4)

args = parser.parse_args()

def add_convolution(m, depth, kernel_size=3, input_shape=[]):
    m.add(Convolution2D(depth, kernel_size, kernel_size, border_mode="same", use_bias=True, input_shape=input_shape))
 #   m.add(BatchNormalization())
    m.add(ELU())


data = []
for d in args.dirs:
    data += load_data(d)

model = Sequential()
add_convolution(model, 4, input_shape=(64,64,1))
model.add(MaxPooling2D())
add_convolution(model, 4)
model.add(MaxPooling2D())
add_convolution(model, 4)
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(16))
model.add(ELU())
model.add(Dense(1))
model.compile(optimizer=Adam(lr=0.01), loss="mse")

data_train, data_val = data[:-args.nval], data[-args.nval:] 
data_val = data_val[::args.mval]

val_gen = DataGenerator(data_val, augment_data=False)
train_gen = DataGenerator(data_train, augment_data=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1,min_lr=1e-7)
model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit_generator(\
  train_gen,\
  samples_per_epoch=len(data_train),\
  nb_epoch=100,\
  validation_data=val_gen,
  nb_val_samples=len(data_val),
  max_q_size=32,
  nb_worker=8,
  pickle_safe=True,
  callbacks=[reduce_lr, model_checkpoint])
