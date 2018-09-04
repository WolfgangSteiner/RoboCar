from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ELU, LeakyReLU, ReLU
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.callbacks import  ReduceLROnPlateau,EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers, metrics
from sklearn.model_selection import train_test_split
from Common import load_data, load_pickled_data
from DataGenerator import DataGenerator
import click
import random
import os


ROBOT = os.environ["ROBOT"]

def regularizer(r):
    if r == 0.0:
        return None
    else:
        return regularizers.l2(r)


def get_activation(s):
    if s == "elu":
        return ELU()
    elif s == "relu":
        return ReLU()
    elif s == "leaky_relu":
        return LeakyReLU()


def add_convolution(m, depth, kernel_size=3, input_shape=[], regularization=0.0, activation="elu", batch_normalization=False):
    m.add(Conv2D(
        depth, 
        (kernel_size, kernel_size),
        padding="same",
        use_bias=True,
        input_shape=input_shape,
        kernel_regularizer=regularizer(regularization)))
    if batch_normalization:
        m.add(BatchNormalization())
    m.add(get_activation(activation))


@click.command()
@click.argument('dirs', nargs=-1, type=click.Path(exists=True))
@click.option('--conv', type=str, default="8-8-8", show_default=True)
@click.option('--dense', type=str, default="32", show_default=True)
@click.option('--conv-regularization', type=float, default=0.0, show_default=True)
@click.option('--dense-regularization', type=float, default=0.0, show_default=True)
@click.option('--dense-dropout', type=float, default=0.0, show_default=True)
@click.option('--learning-rate', type=float, default=0.001, show_default=True)
@click.option('--nval', type=int, default=1024, show_default=True)
@click.option('--mval', type=int, default=4, show_default=True)
@click.option('--batch-size', type=int, default=32, show_default=True)
@click.option('--num-epochs', type=int, default=100, show_default=True)
@click.option('--samples-per-epoch', type=int, default=-1, show_default=True)
@click.option('--activation', type=click.Choice(['elu', 'relu', 'leaky_relu']), default='elu', show_default=True)
@click.option('--batch-normalization/--no-batch-normalization', default=True, show_default=True)
@click.option('--scale', type=int, default=1, show_default=True)
@click.option('--crop-bottom', type=int, default=0, show_default=True)
@click.option('--crop-top', type=int, default=0, show_default=True)
@click.option('--summary', is_flag=True, show_default=True) 
def train_model(dirs, conv, dense, conv_regularization, dense_regularization, learning_rate, nval, mval, batch_size, num_epochs, samples_per_epoch, dense_dropout, activation, batch_normalization, scale, crop_top, crop_bottom, summary):
    #conv = [int(i) for i in conv.split(',')]
    dense = [int(_) for _ in dense.split(',')]
    input_shape = (64 // scale - crop_top - crop_bottom, 64 // scale, 1)
    model = Sequential()
    
    for i in conv.split('-'):
        if i != '':
            for j in i.split(','):
                j = [int(k) for k in j.split('x')]
                if len(j) == 1:
                    j.append(3)
                (depth, kernel_size) = j
                click.echo(f"Conv {kernel_size}x{kernel_size}x{depth}")
                add_convolution(model, depth, input_shape=input_shape, regularization=conv_regularization, activation=activation, batch_normalization=batch_normalization)
                input_shape = []
 
        click.echo(f"MaxPooling")
        model.add(MaxPooling2D())
        input_shape = []

    model.add(Flatten())
    
    for n in dense:
        if n == 0:
            continue
        
        if dense_dropout > 0.0:
            click.echo(f"Dropout {dense_dropout}")
            model.add(Dropout(dense_dropout))

        click.echo(f"Dense {n}")
        model.add(Dense(n, kernel_regularizer=regularizer(dense_regularization)))
        model.add(get_activation(activation))
    
    model.add(Dense(1))

    click.echo("Compiling model...")
    model.compile(optimizer=Adam(lr=learning_rate), loss="mse", metrics=[metrics.mse])
    
    if summary:
        print(model.summary())

    click.echo("Loading data...")
    data = []
    if not len(dirs):
        dirs = [f'/home/wolfgang/RoboCar/data.{ROBOT}']
    for d in dirs:
        data += load_pickled_data(d)
    click.echo(f"{len(data)} training images")
    
    data_train, data_val = data[:-nval], data[-nval:]
    if samples_per_epoch == -1:
        samples_per_epoch = len(data_train)

    random.seed(42)
    random.shuffle(data_train)

    data_val = data_val[::mval]

    crop_x = [0,0]
    crop_y=[crop_top, crop_bottom]
    val_gen = DataGenerator(data_val, augment_data=False, scale=scale, crop_x=crop_x, crop_y=crop_y)
    train_gen = DataGenerator(data_train, batch_size=batch_size, augment_data=True, scale=scale, crop_x=crop_x, crop_y=crop_y)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1,min_lr=1e-7)
    
    model_checkpoint = ModelCheckpoint(
      f"model.{ROBOT}.h5", 
      monitor='val_loss', 
      verbose=1,
      save_best_only=True,
      save_weights_only=False,
      mode='auto',
      period=1)
     
    model.fit_generator(
      train_gen,
      samples_per_epoch=samples_per_epoch,
      nb_epoch=num_epochs,
      validation_data=val_gen,
      nb_val_samples=len(data_val),
      max_q_size=32,
      nb_worker=8,
      pickle_safe=True,
      callbacks=[reduce_lr, model_checkpoint])


if __name__ == "__main__":
    train_model()
