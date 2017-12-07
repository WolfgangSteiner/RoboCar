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
args = parser.parse_args()

def add_convolution(m, depth, kernel_size=3):
    m.add(Convolution2D(depth, kernel_size, kernel_size, kernel_regularizer=regularizers.l2(0.01), border_mode="same", use_bias=False))
    m.add(BatchNormalization())
    m.add(ELU())


data = []
for d in args.dirs:
    data += load_data(d)

model = Sequential()
model.add(AveragePooling2D((8,8), input_shape=(480,640,3)))
add_convolution(model, 1, 1)
add_convolution(model, 16)
model.add(MaxPooling2D())
add_convolution(model, 32)
model.add(MaxPooling2D())
add_convolution(model, 64)
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(ELU())
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))
model.compile(optimizer=Adam(lr=0.0001), loss="mse")

data_train, data_val = train_test_split(data, test_size=256, random_state=42)
val_gen = DataGenerator(data_val, augment_data=False)
train_gen = DataGenerator(data_train, augment_data=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1,min_lr=1e-7)
model_checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model.fit_generator(\
  train_gen,\
  samples_per_epoch=2**11,\
  nb_epoch=100,\
  validation_data=val_gen,
  nb_val_samples=len(data_val),
  max_q_size=32,
  nb_worker=8,
  pickle_safe=True,
  callbacks=[reduce_lr, model_checkpoint])
