#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:50:23 2019

@author: detlef
"""
#pylint: disable=R0903, C0301, C0103, C0111

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# restarting in the same console throws an tensorflow error, force a new console
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from random import shuffle
from random import random, randint

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model, Input
from keras.layers import Activation, Embedding, Dense, Flatten, GlobalMaxPooling1D, GlobalAveragePooling1D, Lambda
from keras.layers import LSTM, CuDNNLSTM, CuDNNGRU, SimpleRNN, GRU
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.callbacks import ModelCheckpoint
import keras.backend

# uncomment the following to disable CuDNN support
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#LSTM_use = LSTM
###########################################


import argparse
from random import shuffle

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-3)
parser.add_argument('--epochs', dest='epochs',  type=int, default=50)
parser.add_argument('--hidden_size', dest='hidden_size',  type=int, default=50)
parser.add_argument('--final_name', dest='final_name',  type=str, default='final_model')
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)
parser.add_argument('--attention', dest='attention', action='store_true')
#parser.add_argument('--depth', dest='depth',  type=int, default=3)
parser.add_argument('--debug', dest='debug', action='store_true')
#parser.add_argument('--only_one', dest='only_one', action='store_true')
parser.add_argument('--revert', dest='revert', action='store_true')
#parser.add_argument('--add_history', dest='add_history', action='store_true')
parser.add_argument('--RNN_type', dest='RNN_type',  type=str, default='CuDNNLSTM')
parser.add_argument('--gpu_mem', dest='gpu_mem',  type=float, default=0.5)
#parser.add_argument('--fill_vars_with_atoms', dest='fill_vars_with_atoms', action='store_true')
#parser.add_argument('--rand_atoms', dest='rand_atoms', action='store_true')
parser.add_argument('--float_type', dest='float_type',  type=str, default='float32')
parser.add_argument('--epoch_size', dest='epoch_size',  type=int, default=100000)

args = parser.parse_args()

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
set_session(tf.Session(config=config))

keras.backend.set_floatx(args.float_type)



RNN_type = {}
RNN_type['CuDNNLSTM'] = CuDNNLSTM
RNN_type['CuDNNGRU'] = CuDNNGRU
RNN_type['GRU'] = GRU
RNN_type['SimpleRNN'] = SimpleRNN

LSTM_use = RNN_type[args.RNN_type]

vocab = {}
vocab_rev = {}
count_chars = 0
def add_translate(cc):
    #pylint: disable=W0603
    global count_chars
    vocab[cc] = count_chars
    vocab_rev[count_chars] = cc
    count_chars += 1

#for c in range(ord('a'), ord('z')+1):
#    add_translate(chr(c))
for c in range(ord('0'), ord('9')+1):
    add_translate(chr(c))

add_translate('+')
add_translate('*')
add_translate('=')
add_translate('.')
add_translate(' ')

print("num of different chars", len(vocab))

def check_all_chars_in(x):
    for cc in x:
        if cc not in vocab:
            return False
    return True

print(vocab)

max_output = len(vocab)
###################################################################
# Network


def attentions_layer(x):
  from keras import backend as K
  x1 = x[:,:,1:]
  x2 = x[:,:,0:1]
  x2 = K.softmax(x2)
#  x2 = keras.backend.print_tensor(x2, str(x2))
#  x1 = keras.backend.print_tensor(x1, str(x1))
  x=x1*x2
#  x = keras.backend.print_tensor(x, str(x))
  return x

hidden_size = args.hidden_size

if args.pretrained_name is not None:
  from keras.models import load_model
  model = load_model(args.pretrained_name)
  print("loaded model",model.layers[0].input_shape[1])
#  ml = model.layers[0].input_shape[1]
#  if (ml != max_length):
#    print("model length",ml,"different from data length",max_length)
#    max_length = ml
else:
#  model = Sequential()
#  model.add(Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=False, input_shape=(max_length,)))
#  model.add(LSTM_use(hidden_size, return_sequences=True))
#  model.add(LSTM_use(max_output + 1, return_sequences=False))
#  model.add(Dense(max_output +1))
#  model.add(Activation('softmax'))
  
  inputs = Input(shape=(None,))
  embeds = Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=True)(inputs)
  lstm1 = LSTM_use(hidden_size, return_sequences=True)(embeds)
  if args.attention:
    lstm1b = Lambda(attentions_layer)(lstm1)
  else:
    lstm1b = lstm1
  lstm4 = LSTM_use(hidden_size, return_sequences=True)(lstm1b)
#  x1 = Dense(hidden_size, activation='relu')(lstm4)
#  x2 = Dense(hidden_size, activation='relu')(x1)
#  x3 = Dense(hidden_size, activation='relu')(x2)
  x = Dense(max_output)(lstm4)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)




import inspect
with open(__file__) as f:
    a = f.readlines()
startline = inspect.currentframe().f_lineno
print(a[startline+1:startline+2])
optimizer = RMSprop(lr=args.lr, rho=0.9, epsilon=None, decay=0)

print("learning rate",keras.backend.eval(optimizer.lr))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

print(model.summary())

max_length = -1

def str_to_int_list(x, ml):
    # uncomment for reverse
    if args.revert:
      x = x[::-1]
    # uncomment for all the same length
    #x = ('{:>'+str(ml)+'}').format(x[-ml:])
    ret = []
    for cc in x:
        ret.append(vocab[cc])
    return ret

class KerasBatchGenerator(object):

    def __init__(self, datain, vocabin):
        self.data = datain
        self.vocab = vocabin
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        # self.current_idx = 0

    def generate(self):
        while True:
            #create the string
            xx1 = [randint(0,9) for _ in range(2)]
            x1 = 0
            for x in xx1:
               x1*=10
               x1+=x
            xx2 = [randint(0,9) for _ in range(8)]
            x2 = 0
            for x in xx2:
               x2*=10
               x2+=x
            #x2 = randint(0,999)
            rr = x1*x2
            zw = ""
            faktor = 1
            for x in xx1[::-1]:
                zw += str(x*x2*faktor)+"+"
                faktor *=10
            zw = zw[:-1][::-1]
            #print (zw)
            inn = ((str(x1) + "*" + str(x2))[::-1] + ("=" * 3)) * 1 + ((" " * len(zw))    + (" " * 1)) * 1  + (" " * len(str(rr))) * 0 + "."
            out = ((str(x1) + "*" + str(x2))[::-1] + ("=" * 3)) * 1 + (zw                 + ("=" * 1)) * 1  + str(rr)[::-1]        * 0 + "."
            
            tmp_x = np.array([str_to_int_list(inn, max_length)], dtype=int).reshape((1, -1))
            
            tmp_y = np.array([str_to_int_list(out, max_length)], dtype=int).reshape((1, -1))
            
            yield tmp_x, to_categorical(tmp_y, num_classes=max_output)


train_data_generator = KerasBatchGenerator("train", vocab)
valid_data_generator = KerasBatchGenerator("valid", vocab)


print("starting")
checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

num_epochs = args.epochs

history = model.fit_generator(train_data_generator.generate(), args.epoch_size, num_epochs, validation_data=valid_data_generator.generate(), validation_steps=args.epoch_size / 10, callbacks=[checkpointer])

model.save(args.final_name+'.hdf5')
print(history.history.keys())

def list_to_string(prediction):
    s=""
    for i in range(prediction.shape[0]):
        s += vocab_rev[np.argmax(prediction[i])]
    return s
    
    
sum_correct = 0
ccc = 0
for inn,out in valid_data_generator.generate():
    prediction = model.predict(inn)[0]
    o_str = list_to_string(out[0])
    p_str = list_to_string(prediction)
    if o_str == p_str:
        sum_correct+=1
    else:
        print(o_str, p_str)
    ccc +=1
    if ccc >=100:
        print("correct: "+str(sum_correct)+"/"+str(ccc)+"="+str(sum_correct/ccc))
        break
