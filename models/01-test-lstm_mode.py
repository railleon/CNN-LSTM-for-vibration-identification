from tensorflow.keras.models import Model
import tensorflow.keras.layers
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.regularizers import l2

from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable
# from utils.layer_utils import AttentionLSTM

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, MaxPool2D, MaxPooling1D
from tensorflow.keras.optimizers import Adam           # 优化器
# Adam=adam_v2.Adam
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DATASET_INDEX = 49

MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True

regularization_weight = 5e-4


def generate_model_CNNLSTM4():
    ip = keras.layers.Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    y = keras.layers.LSTM(64)(ip)
    y = Dropout(0.5)(y)

    x = keras.layers.Reshape((MAX_NB_VARIABLES, MAX_TIMESTEPS,1))(ip)


    x = Conv2D(32, (3, 3), padding='same',
               kernel_initializer='he_normal', name='conv2d_1')(x)
    x = Activation('relu', name='activation_1')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same',
               kernel_initializer='he_normal', name='conv2d_2')(x)
    x = Activation('relu', name='activation_2')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same',
               kernel_initializer='he_normal', name='conv2d_3')(x)
    x = Activation('relu', name='activation_3')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = keras.layers.concatenate([x, y])

    x = keras.layers.Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    out = keras.layers.Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()

    return model



def outputtxt(accx,valaccx,lossx,vallossx):
    plt.figure(1)
    plt.plot(accx, "g--", label="Accuracy of training data")
    plt.plot(valaccx, "g", label="Accuracy of validation data")
    plt.plot(lossx, "r--", label="Loss of training data")
    plt.plot(vallossx, "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()

    plt.figure(1).savefig(r'acc__'+num+'.jpg')

    plt.close('all')

    acc=np.array(accx)
    np.savetxt('acc__'+num+'.txt', acc)

    acc=np.array(valaccx)
    np.savetxt('val_acc__'+num+'.txt', acc)

    acc=np.array(lossx)
    np.savetxt('loss__'+num+'.txt', acc)

    acc=np.array(vallossx)
    np.savetxt('val_loss__'+num+'.txt', acc)


if __name__ == "__main__":

    epochs=1000
    batch_size=128


    num = 'CNNLSTM4'
    start1 = time.time()
    model = generate_model_CNNLSTM4()

    accuracy, loss, pre_label, ture_label = evaluate_model(model, DATASET_INDEX,
                                                           dataset_prefix=num,
                                                           batch_size=2,
                                                           predict=True)




    with open('accuracy.txt', 'a') as file0:
        print(num, '_accuracy:' % (accuracy), file=file0)
#####################################################################