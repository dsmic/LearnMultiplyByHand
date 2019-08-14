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

from random import randint

import numpy as np
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Activation, Embedding, Lambda, Concatenate, Layer, Dense, Dropout, GaussianNoise
from keras.layers import CuDNNLSTM, CuDNNGRU, SimpleRNN, GRU
from keras.optimizers import  RMSprop
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import keras

# uncomment the following to disable CuDNN support
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#LSTM_use = LSTM
###########################################


import argparse

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-3)
parser.add_argument('--epochs', dest='epochs',  type=int, default=50)
parser.add_argument('--hidden_size', dest='hidden_size',  type=int, default=50)
parser.add_argument('--lstm_num', dest='lstm_num',  type=int, default=3)
parser.add_argument('--final_name', dest='final_name',  type=str, default='final_model')
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)
parser.add_argument('--attention', dest='attention', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--embed_not_trainable', dest='embed_not_trainable', action='store_true')
parser.add_argument('--revert', dest='revert', action='store_true')
#parser.add_argument('--add_history', dest='add_history', action='store_true')
parser.add_argument('--RNN_type', dest='RNN_type',  type=str, default='CuDNNLSTM')
parser.add_argument('--gpu_mem', dest='gpu_mem',  type=float, default=0.5)
parser.add_argument('--float_type', dest='float_type',  type=str, default='float32')
parser.add_argument('--epoch_size', dest='epoch_size',  type=int, default=100000)
parser.add_argument('--train_data_num', dest='train_data_num',  type=int, default=1000)
parser.add_argument('--use_two_LSTM', dest='use_two_LSTM', action='store_true')
parser.add_argument('--load_weights_name', dest='load_weights_name',  type=str, default=None)
parser.add_argument('--check_data_num', dest='check_data_num',  type=int, default=1000)
parser.add_argument('--selector_pow', dest='selector_pow',  type=int, default=1)
parser.add_argument('--dropout', dest='dropout',  type=float, default=0.0)
parser.add_argument('--noise_layer', dest='noise_layer',  type=float, default=0.0)
parser.add_argument('--start_console', dest='start_console', action='store_true')
parser.add_argument('--select_without_dense', dest='select_without_dense', action='store_true')
parser.add_argument('--use_full_select_layer', dest='use_full_select_layer', action='store_true')

args = parser.parse_args()

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
set_session(tf.Session(config=config))

K.set_floatx(args.float_type)



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

for c in range(ord('a'), ord('l')+1):
    add_translate(chr(c))
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
#  from keras import backend as K
  x1 = x[:,:,1:]
  x2 = x[:,:,0:1]
  x2 = K.softmax(x2)
#  x2 = keras.backend.print_tensor(x2, str(x2))
#  x1 = keras.backend.print_tensor(x1, str(x1))
  x=x1*x2
#  x = keras.backend.print_tensor(x, str(x))
  return x

@tf.custom_gradient
def select_subnet_layer(x):
    size_of_out = (x.shape[2]-args.lstm_num) // args.lstm_num
    oo = [x[:,:,(i * size_of_out):((i+1)*size_of_out)]* x[:,:,size_of_out * args.lstm_num +i:size_of_out * args.lstm_num + i + 1] for i in range(args.lstm_num)]
    def custom_grad(dy):
        gg = [tf.pow(x[:,:,size_of_out * args.lstm_num +i:size_of_out * args.lstm_num + i + 1],args.selector_pow) * dy 
              for i in range(args.lstm_num)]
        gs = [K.sum(x[:,:,size_of_out * i:size_of_out * (i+1)] * dy, axis = 2, keepdims = True) 
                for i in range(args.lstm_num)]
        grad = K.concatenate(gg + gs)
        return grad
    return tf.add_n(oo), custom_grad

class SelectSubnetLayer2(Layer):

#    def __init__(self, **kwargs):
#        super(CustomLayer, self).__init__(**kwargs)

    def call(self, x):
        return select_subnet_layer(x)  # you don't need to explicitly define the custom gradient

    def compute_output_shape(self, input_shape):
        print(input_shape[2])
        return (input_shape[0], input_shape[1], (int(input_shape[2]) - args.lstm_num ) // args.lstm_num)

class SelectSubnetLayer(Layer):
    # this layer takes as input a list of identically shaped inputs and outputs the shape of one of thouse inputs without the first entry in the last axis
    # the first entry in the last axis is used as a selector. From all selectors with softmax every input is scaled and added to result in the output
    # The idea is to select one subnet for a given task. Different subnets can do different task and the selector selects, which subnet is used.
    # the layer supports dropout for the selector (before softmax)
    def __init__(self, dropout = 0, **kwargs):
        self.dropout = dropout
        super(SelectSubnetLayer, self).__init__(**kwargs)

    def call(self, x, training=None):
        sel = [xx[:,:,0:1] for xx in x] #the first enty of every input is the selector
        sel_tensor = K.concatenate(sel)
        sel_drop = K.dropout(sel_tensor,self.dropout) #drop out of selector before softmax
        self.sel_drop_softmax = K.softmax(K.in_train_phase(sel_drop,sel_tensor, training=training))
        
        oo = [x[i][:,:,1:]*self.sel_drop_softmax[:,:,i:i+1] for i in range(len(x))]
        
        return [tf.add_n(oo),  self.sel_drop_softmax] # you don't need to explicitly define the custom gradient
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        f_input_shape = input_shape[0]
        print(f_input_shape[2])
        return [(f_input_shape[0], f_input_shape[1], f_input_shape[2] - 1), (f_input_shape[0], f_input_shape[1], len(input_shape))]
    
hidden_size = args.hidden_size

if args.pretrained_name is not None:
  from keras.models import load_model
  model = load_model(args.pretrained_name, custom_objects = { "keras": keras , "args":args})
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
  
  inputs = Input(shape=(None,None))
  print("inputs",inputs.shape)
  x0 = Lambda(lambda x : x[:,0,:])(inputs)
  x1 = Lambda(lambda x : x[:,1,:])(inputs)
  x2 = Lambda(lambda x : x[:,2,:])(inputs)
  x3 = Lambda(lambda x : x[:,3,:])(inputs)

  embeds0 = Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=not args.embed_not_trainable)(x0)
  embeds1 = Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=not args.embed_not_trainable)(x1)
  embeds2 = Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=not args.embed_not_trainable)(x2)
  embeds3 = Embedding(len(vocab), len(vocab), embeddings_initializer='identity', trainable=not args.embed_not_trainable)(x3)

  print("x0",x0.shape)
  conc = Concatenate()([embeds0,embeds1,embeds2,embeds3])#,embeds4,embeds5])#,embeds6,embeds7,embeds8,embeds9])
  lstm4 = []
  lstm4_select = []
  lstm2_list = []
  for _ in range(args.lstm_num):
      if not args.use_two_LSTM:
          lstm1 = conc
      else:
          lstm1 = LSTM_use(hidden_size, return_sequences=True)(conc)
      if args.attention:
        lstm1b = Lambda(attentions_layer)(lstm1)
      else:
        lstm1b = lstm1
      lstm_2 = LSTM_use(hidden_size, return_sequences=True)(lstm1b)  
      lstm2_list.append(lstm_2)
      if args.select_without_dense:
          lstm2b_select = Lambda(lambda x: x[:,:,0:1])(lstm_2)
          lstm2b_data = Lambda(lambda x: x[:,:,1:])(lstm_2)
          lstm4_select.append(lstm2b_select)
      else:
          lstm2b_data = lstm_2
      lstm4.append(lstm2b_data)
  if not args.use_full_select_layer:
      if args.lstm_num > 1:
          cc = Concatenate()(lstm4)
      else:
          cc = lstm4[0]
      if not args.select_without_dense:
          s_select = Dense(args.lstm_num, #activation='softmax', 
                       name='dense_selector_'+str(args.lstm_num))(cc)
      else:
          s_select = Concatenate()(lstm4_select)
      if args.dropout > 0:
        s_select_drop = Dropout(rate = args.dropout)(s_select)
      else:
        s_select_drop = s_select  
      if args.noise_layer > 0:
          s_select_drop_noise = GaussianNoise(args.noise_layer)(s_select_drop)
      else:
        s_select_drop_noise = s_select_drop  
      s_select_drop_activation = Activation('softmax', name = 'dense_selector_act')(s_select_drop_noise)
      cc2 = Concatenate()([cc,s_select_drop_activation])
      ccc = SelectSubnetLayer2()(cc2)
  else:
      cccb = SelectSubnetLayer(args.dropout)(lstm2_list)
      ccc = Concatenate()(cccb)
  x = Dense(max_output)(ccc)
  predictions = Activation('softmax')(x)
  model = Model(inputs=inputs, outputs=predictions)



import inspect
with open(__file__) as f:
    a = f.readlines()
startline = inspect.currentframe().f_lineno
print(a[startline+1:startline+2])
optimizer = RMSprop()

print("learning rate",K.eval(optimizer.lr))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

print(model.summary(line_length=180, positions = [.33, .55, .67, 1.]))

if args.load_weights_name:
    model.load_weights(args.load_weights_name, by_name=True)

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

def n_str_to_int_list(x, ml):
    # uncomment for reverse
    if args.revert:
      x = x[::-1]
    # uncomment for all the same length
    #x = ('{:>'+str(ml)+'}').format(x[-ml:])
    ret = []
    for cc in x:
        r1 = []
        for ccc in cc:
            r1.append(vocab[ccc])
        ret.append(r1)
    return ret

def one_data(maxlen1, maxlen2, debug = False):
    sequence_out = []
    sequence_in = []
    result = []
    
    xx1 = [randint(0,9) for _ in range(maxlen1)]
    x1 = 0
    for x in xx1:
       x1*=10
       x1+=x
    xx2 = [randint(0,9) for _ in range(maxlen2)]
    x2 = 0
    for x in xx2:
       x2*=10
       x2+=x
    if debug:
        print(x1*x2)
    
    len1 = len(str(x1))
    len2 = len(str(x2))
    
    if debug:
        print(len1,len2)
    
    result.append(str(x1)+"*"+str(x2)+"=")
    
    pos1 = [0,0]
    pos2 = [0,0]
    pos3 = [0,0]
    
    def movepos(p,d):
        if d == 'r':
            p[0]+=1
        elif d == 'l':
            p[0]-=1
        elif d == 'u':
            p[1]-=1
        elif d == 'd':
            p[1]+=1
#        if p[0]<0:
#            p[0]=0
#        if p[1]<0:
#            p[1]=0
    
    def get_int1():
        if debug:
            print("pos1",pos1,get_vector_pos(pos1),get_vector_pos(pos2))
        if pos1[1]<0:
            return 0
        r = result[pos1[1]][pos1[0]]
        if r == ' ':
            return 0
        return int(r)
    def get_int2():
        if debug:
            print("pos2",pos2,get_vector_pos(pos1),get_vector_pos(pos2))
        if len(result[pos2[1]]) <= pos2[0]:
            return 0
        r = result[pos2[1]][pos2[0]]
        if r == ' ':
            return 0
        return int(r)
    def get_int3():
        if debug:
            print("pos3",pos3)
        r = result[pos3[1]][pos3[0]]
        if r == ' ':
            return 0
        return int(r)
    
    def get_vector_pos(pos):
        vec = []
        if pos[1]>=0 and pos[1]<len(result)-1 and pos[0]<len(result[pos[1]]):
            vec.append(result[pos[1]][pos[0]])
        else:
            vec.append(' ')
#        if pos[1]>0 and pos[0]<len(result[pos[1]-1]):
#            vec.append(result[pos[1]-1][pos[0]])
#        else:
#            vec.append(' ')
#        if pos[1]>=0 and pos[1]<len(result)-1 and pos[0]<len(result[pos[1]+1]):
#            vec.append(result[pos[1]+1][pos[0]])
#        else:
#            vec.append(' ')
#        if pos[0]>0 and pos[0] <len(result[pos[1]]):
#            vec.append(result[pos[1]][pos[0]])
#        else:
#            vec.append(' ')
        if pos[1]>=0 and pos[0] + 1<len(result[pos[1]]):
            vec.append(result[pos[1]][pos[0]+1])
        else:
            vec.append(' ')
        return vec
        
           
    def move_or_set(direction):
        #global sequence_out
        #global seqence_in
        
        sequence_out.append(direction)
        sequence_in.append(get_vector_pos(pos1)+get_vector_pos(pos2))
        if direction == 'a':
            movepos(pos1,'r')
        elif direction == 'b':
            movepos(pos1,'l')
        elif direction == 'c':
            movepos(pos1,'d')
        elif direction == 'd':
            movepos(pos1,'u')
    
        elif direction == 'e':
            movepos(pos2,'r')
        elif direction == 'f':
            movepos(pos2,'l')
        elif direction == 'g':
            movepos(pos2,'d')
        elif direction == 'h':
            movepos(pos2,'u')
    
        elif direction == 'i':
            movepos(pos3,'r')
        elif direction == 'j':
            movepos(pos3,'l')
        elif direction == 'k':
            movepos(pos3,'d')
        elif direction == 'l':
            movepos(pos3,'u')
        else:
            while pos3[1] >= len(result):
                result.append("")
            while pos3[0] >= len(result[pos3[1]]):
                result[pos3[1]] += ' '
            result[pos3[1]] = result[pos3[1]][:pos3[0]]+direction+result[pos3[1]][pos3[0]+1:]
        #double in case of set but not move
        while pos3[1] >= len(result):
            result.append("")
        while pos3[0] >= len(result[pos3[1]]):
            result[pos3[1]] += ' '
    
    #move_or_set('a')
    #move_or_set('e')
    #move_or_set('i')
    
    for _ in range(len1-1):
        move_or_set('a')
    for _ in range(len1+2-1):
        move_or_set('e')
        move_or_set('i')
    move_or_set('k')
    
    for pp in range(len2):
        p = get_int1()*get_int2()
        c = int(p / 10)
        move_or_set(str(p)[-1:])
        if debug:
                for r in result:
                    print(r)
        for _ in range(len1-1):
            move_or_set('b')
            move_or_set('j')
            p = get_int1()*get_int2()+c
            c = int(p / 10)
            move_or_set(str(p)[-1:])
            if debug:
                for r in result:
                    print(r)
        move_or_set('j')
        move_or_set(str(c))
        move_or_set('i')
        for _ in range(len1-1):
            move_or_set('a')
            move_or_set('i')
        if pp < len2-1:
            move_or_set('i')
            move_or_set('e')
        move_or_set('k')
    move_or_set('g')
    move_or_set('d') # to be different for sum
    for pp in range(len1+len2):
        move_or_set('k')
        if debug:
                print(pos1,pos2,pos3)
        p=0
        for _ in range(len2+1):
            p += get_int2()
            move_or_set('g')
        move_or_set(str(p)[-1:])
        c = int(p / 10)
        move_or_set('j')
        if pp < len1+len2-1:
            move_or_set('l')
            move_or_set(str(c))
            if debug:
                for r in result:
                    print(r)

            for _ in range(len2+1):
                move_or_set('h')    
            move_or_set('f')
    move_or_set(' ') #end marker, spaces should never be written in other situations
    return sequence_in, sequence_out, result

class KerasBatchGenerator(object):

    def __init__(self, datain, vocabin):
        self.data = datain
        self.vocab = vocabin
        self.current_idx = 0
        self.inn = []
        if int(self.data) > 0:
            print("train data number",self.data)
            for i in range(self.data):
                self.inn.append(one_data(5, 5))
            
    def generate(self):
        while True:
            if self.data == 0:
                inn, out, _ = one_data(5, 5)
            else:
                if self.current_idx >= self.data:
                    self.current_idx = 0
                inn, out, _ = self.inn[self.current_idx]
                self.current_idx += 1
            tmp_x = np.swapaxes(np.array([n_str_to_int_list(inn, max_length)], dtype=int),1,2)
            tmp_y = np.array([str_to_int_list(out, max_length)], dtype=int).reshape((1, -1))
            #print(tmp_x.shape, tmp_y.shape)
            
            yield tmp_x, to_categorical(tmp_y, num_classes=max_output)


train_data_generator = KerasBatchGenerator(args.train_data_num, vocab)
valid_data_generator = KerasBatchGenerator(0, vocab)


print("starting",args)
checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

num_epochs = args.epochs

history = model.fit_generator(train_data_generator.generate(), args.epoch_size, num_epochs, validation_data=valid_data_generator.generate(), validation_steps=args.epoch_size / 10, callbacks=[checkpointer]) 

model.save(args.final_name+'.hdf5')
model.save_weights(args.final_name+'-weights.hdf5')
print(history.history.keys())

def list_to_string(prediction):
    s=""
    for i in range(prediction.shape[0]):
        s += vocab_rev[np.argmax(prediction[i])]
    return s
    
    
def print_test(num_of_tests):
    sum_correct = 0
    ccc = 0
    for inn,out in valid_data_generator.generate():
        inp = model.input
        if not args.use_full_select_layer:
            selection_layer = [l.output for l in model.layers if l.name.startswith('dense_selector_a')]
            functors = [K.function([inp], [out]) for out in selection_layer]
        else:
            selection_layer = [l.output[1] for l in model.layers if isinstance(l,SelectSubnetLayer)]
            functors = [K.function([inp], [out]) for out in selection_layer]
        
        #print(functors)
        if len(functors)>0:
            ds_out = [func([inn]) for func in functors][0][0][0]
            pp=0
            for ab in ds_out:
                if pp % 10 == 0:
                    print(pp)
                pp +=1
                nnn = chr(ord('A') + np.argmax(ab))
                print('{0}-{1:3.2f}'.format(nnn,max(ab)), end = '  ')
            print('')
        
        prediction = model.predict(inn)[0]
        o_str = list_to_string(out[0])
        p_str = list_to_string(prediction)
        if o_str == p_str:
            sum_correct+=1
        else:
            print(o_str, p_str)
        ccc +=1
        if ccc >=num_of_tests:
            print("correct: "+str(sum_correct)+"/"+str(ccc)+"="+str(sum_correct/ccc))
            break

print_test(args.check_data_num)
if args.start_console:
    import code
    code.interact(local=locals())