##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@nus.edu.sg
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory 
## of https://github.com/y2l/mini-imagenet-tools
##
## This file is modified for tensorflow.keras usage by D. Schmicker
##
## original file from https://github.com/y2l/mini-imagenet-tools
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import random
import numpy as np
from tqdm import trange
import imageio


import argparse
parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)
parser.add_argument('--dataset', dest='dataset',  type=str, default='train')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-3)
parser.add_argument('--epochs', dest='epochs',  type=int, default=10)
parser.add_argument('--final_name', dest='final_name',  type=str, default='final_model')
parser.add_argument('--shuffle_images', dest='shuffle_images', action='store_true')
parser.add_argument('--enable_idx_increase', dest='enable_idx_increase', action='store_true')
parser.add_argument('--use_independent_base', dest='use_independent_base', action='store_true')
parser.add_argument('--train_indep_and_dependent', dest='train_indep_and_dependent', action='store_true')
parser.add_argument('--tensorboard_log_dir', dest='tensorboard_log_dir',  type=str, default='./logs')
args = parser.parse_args()

class MiniImageNetDataLoader(object):
    def __init__(self, shot_num, way_num, episode_test_sample_num):
        self.shot_num = shot_num
        self.way_num = way_num
        self.episode_test_sample_num = episode_test_sample_num
        self.num_samples_per_class = episode_test_sample_num + shot_num
        metatrain_folder = './processed_images/train'
        metaval_folder = './processed_images/val'
        metatest_folder = './processed_images/test'

        npy_dir = './episode_filename_list/'
        if not os.path.exists(npy_dir):
            os.mkdir(npy_dir)

        self.npy_base_dir = npy_dir + str(self.shot_num) + 'shot_' + str(self.way_num) + 'way_' + str(episode_test_sample_num) + 'shuffled_' + str(args.shuffle_images) + '/'
        if not os.path.exists(self.npy_base_dir):
            os.mkdir(self.npy_base_dir)

        self.metatrain_folders = [os.path.join(metatrain_folder, label) \
            for label in os.listdir(metatrain_folder) \
            if os.path.isdir(os.path.join(metatrain_folder, label)) \
            ]
        self.metaval_folders = [os.path.join(metaval_folder, label) \
            for label in os.listdir(metaval_folder) \
            if os.path.isdir(os.path.join(metaval_folder, label)) \
            ]
        self.metatest_folders = [os.path.join(metatest_folder, label) \
            for label in os.listdir(metatest_folder) \
            if os.path.isdir(os.path.join(metatest_folder, label)) \
            ]

    def get_images(self, paths, labels, nb_samples=None, shuffle=True):
        if nb_samples is not None:
            sampler = lambda x: random.sample(x, nb_samples)
        else:
            sampler = lambda x: x
        images = [(i, os.path.join(path, image)) \
            for i, path in zip(labels, paths) \
            for image in sampler(os.listdir(path))]
        if shuffle:
            random.shuffle(images)
        return images

    def generate_data_list(self, phase='train', episode_num=None):
        if phase=='train':
            folders = self.metatrain_folders
            if episode_num is None:
                episode_num = 20000
            if not os.path.exists(self.npy_base_dir+'/train_filenames.npy'):
                print('Generating train filenames')
                all_filenames = []
                for _ in trange(episode_num):
                    sampled_character_folders = random.sample(folders, self.way_num)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num), nb_samples=self.num_samples_per_class, shuffle=args.shuffle_images)
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                np.save(self.npy_base_dir+'/train_labels.npy', labels)
                np.save(self.npy_base_dir+'/train_filenames.npy', all_filenames)
                print('Train filename and label lists are saved')

        elif phase=='val':
            folders = self.metaval_folders
            if episode_num is None:
                episode_num = 600
            if not os.path.exists(self.npy_base_dir+'/val_filenames.npy'):
                print('Generating val filenames')
                all_filenames = []
                for _ in trange(episode_num):
                    sampled_character_folders = random.sample(folders, self.way_num)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num), nb_samples=self.num_samples_per_class, shuffle=args.shuffle_images)
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                np.save(self.npy_base_dir+'/val_labels.npy', labels)
                np.save(self.npy_base_dir+'/val_filenames.npy', all_filenames)
                print('Val filename and label lists are saved')
                
        elif phase=='test':
            folders = self.metatest_folders
            if episode_num is None:
                episode_num = 600
            if not os.path.exists(self.npy_base_dir+'/test_filenames.npy'):
                print('Generating test filenames')
                all_filenames = []
                for _ in trange(episode_num):
                    sampled_character_folders = random.sample(folders, self.way_num)
                    random.shuffle(sampled_character_folders)
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num), nb_samples=self.num_samples_per_class, shuffle=args.shuffle_images)
                    labels = [li[0] for li in labels_and_images]
                    filenames = [li[1] for li in labels_and_images]
                    all_filenames.extend(filenames)
                np.save(self.npy_base_dir+'/test_labels.npy', labels)
                np.save(self.npy_base_dir+'/test_filenames.npy', all_filenames)
                print('Test filename and label lists are saved')
        else:
            print('Please select vaild phase')

    def load_list(self, phase='train'):
        if phase=='train':
            self.train_filenames = np.load(self.npy_base_dir + 'train_filenames.npy').tolist()
            self.train_labels = np.load(self.npy_base_dir + 'train_labels.npy').tolist()

        elif phase=='val':
            self.val_filenames = np.load(self.npy_base_dir + 'val_filenames.npy').tolist()
            self.val_labels = np.load(self.npy_base_dir + 'val_labels.npy').tolist()

        elif phase=='test':
            self.test_filenames = np.load(self.npy_base_dir + 'test_filenames.npy').tolist()
            self.test_labels = np.load(self.npy_base_dir + 'test_labels.npy').tolist()

        elif phase=='all':
            self.train_filenames = np.load(self.npy_base_dir + 'train_filenames.npy').tolist()
            self.train_labels = np.load(self.npy_base_dir + 'train_labels.npy').tolist()

            self.val_filenames = np.load(self.npy_base_dir + 'val_filenames.npy').tolist()
            self.val_labels = np.load(self.npy_base_dir + 'val_labels.npy').tolist()

            self.test_filenames = np.load(self.npy_base_dir + 'test_filenames.npy').tolist()
            self.test_labels = np.load(self.npy_base_dir + 'test_labels.npy').tolist()

        else:
            print('Please select vaild phase')

    def process_batch(self, input_filename_list, input_label_list, batch_sample_num, reshape_with_one=True):
        new_path_list = []
        new_label_list = []
        #assert(input_filename_list[0].startswith('./processed_images/test'))
        #print('process batch', input_filename_list[0])
        for k in range(batch_sample_num):
            class_idxs = list(range(0, self.way_num))
            random.shuffle(class_idxs)
            for class_idx in class_idxs:
                true_idx = class_idx*batch_sample_num + k
                new_path_list.append(input_filename_list[true_idx])
                new_label_list.append(input_label_list[true_idx])

        img_list = []
        for filepath in new_path_list:
            this_img = imageio.imread(filepath)
            this_img = this_img / 255.0
            img_list.append(this_img)

        if reshape_with_one:
            img_array = np.array(img_list)
            label_array = self.one_hot(np.array(new_label_list)).reshape([1, self.way_num*batch_sample_num, -1])
        else:
            img_array = np.array(img_list)
            label_array = self.one_hot(np.array(new_label_list)).reshape([self.way_num*batch_sample_num, -1])
        return img_array, label_array

    def one_hot(self, inp):
        n_class = inp.max() + 1
        n_sample = inp.shape[0]
        out = np.zeros((n_sample, n_class))
        for idx in range(n_sample):
            out[idx, inp[idx]] = 1
        return out

    def idx_to_big(self, phase, idx):
        if phase=='train':
            all_filenames = self.train_filenames
#            labels = self.train_labels 
        elif phase=='val':
            all_filenames = self.val_filenames
#            labels = self.val_labels 
        elif phase=='test':
            all_filenames = self.test_filenames
#            labels = self.test_labels
        else:
            print('Please select vaild phase')

        one_episode_sample_num = self.num_samples_per_class*self.shot_num
        return ((idx+1)*one_episode_sample_num >= len(all_filenames))

    def get_batch(self, phase='train', idx=0):
        if phase=='train':
            all_filenames = self.train_filenames
            labels = self.train_labels 
        elif phase=='val':
            all_filenames = self.val_filenames
            labels = self.val_labels 
        elif phase=='test':
            all_filenames = self.test_filenames
            labels = self.test_labels
        else:
            print('Please select vaild phase')

        one_episode_sample_num = self.num_samples_per_class*self.shot_num
        this_task_filenames = all_filenames[idx*one_episode_sample_num:(idx+1)*one_episode_sample_num]
        epitr_sample_num = self.shot_num
        epite_sample_num = self.episode_test_sample_num

        this_task_tr_filenames = []
        this_task_tr_labels = []
        this_task_te_filenames = []
        this_task_te_labels = []
        for class_k in range(self.way_num):
            this_class_filenames = this_task_filenames[class_k*self.num_samples_per_class:(class_k+1)*self.num_samples_per_class]
            this_class_label = labels[class_k*self.num_samples_per_class:(class_k+1)*self.num_samples_per_class]
            this_task_tr_filenames += this_class_filenames[0:epitr_sample_num]
            this_task_tr_labels += this_class_label[0:epitr_sample_num]
            this_task_te_filenames += this_class_filenames[epitr_sample_num:]
            this_task_te_labels += this_class_label[epitr_sample_num:]

        #print(this_task_tr_filenames[0][23:33],this_task_tr_labels[0],this_task_te_filenames[0][23:33],this_task_te_labels[0])
        this_inputa, this_labela = self.process_batch(this_task_tr_filenames, this_task_tr_labels, epitr_sample_num, reshape_with_one=False)
        this_inputb, this_labelb = self.process_batch(this_task_te_filenames, this_task_te_labels, epite_sample_num, reshape_with_one=False)

        return this_inputa, this_labela, this_inputb, this_labelb

cathegories = 5
dataloader = MiniImageNetDataLoader(shot_num=5 * 2, way_num=cathegories, episode_test_sample_num=15) #twice shot_num is because one might be uses as the base for the samples

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

print('mode is',args.dataset)
dataloader.load_list('all')

#print('train',dataloader.train_filenames)
#print('val',dataloader.val_filenames)
#print('test',dataloader.test_filenames)


base_train_img, base_train_label, base_test_img, base_test_label = \
        dataloader.get_batch(phase='train', idx=0) 

train_epoch_size = base_train_img.shape[0]
if not args.train_indep_and_dependent:
    train_epoch_size = int(train_epoch_size / 2) # as double is generated for the base and train
test_epoch_size = base_test_img.shape[0]

print("epoch training size:", train_epoch_size, base_train_label.shape[0], "epoch testing size", test_epoch_size)

class KerasBatchGenerator(object):

#    def __init__(self):

            
    def generate(self, phase='train'):
#        idx = 0
        while True:
#            episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
#                dataloader.get_batch(phase='train', idx=idx)
            if phase == 'train':
                #print(episode_train_img.shape[0])
                for i in range(train_epoch_size):
                    yield base_train_img[i:i+1], base_train_label[i:i+1]
            else:
                #print(episode_test_img.shape[0])
                for i in range(test_epoch_size):
                    yield base_test_img[i:i+1], base_test_label[i:i+1]

    def generate_add_samples(self, phase = 'train'):
        self.idx = 0
        while True:
            batch_train_img, batch_train_label, episode_test_img, episode_test_label = \
                dataloader.get_batch(phase=args.dataset, idx=self.idx)

            # this depends on what we are trying to train.
            # care must be taken, that with a different dataset the labels have a different meaning. Thus if we use a new dataset, we must 
            # use network_base which fits to the database. Therefore there must be taken images with label from the same dataset.
            network_base_img = batch_train_img[:train_epoch_size]
            network_base_img = batch_train_label[:train_epoch_size]
            
            #only half is used now, as the rest is reserved for independend base    
            episode_train_img = batch_train_img[train_epoch_size:]
            episode_train_label = batch_train_label[train_epoch_size:]
            
            if not args.use_independent_base:
                network_base_img = episode_train_img
                network_base_label = episode_train_label
            if args.train_indep_and_dependent:
                episode_train_img = batch_train_img
                episode_train_label = batch_train_label
            if phase == 'train':
                if args.enable_idx_increase:
                    self.idx += 1 # only train phase allowed to change
                    if dataloader.idx_to_big(args.dataset, self.idx):
                        self.idx=0
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("all data used, starting from beginning")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        
                #print(episode_train_img.shape[0])
                #assert(episode_train_img.shape[0] == 25)
                for i in range(train_epoch_size):
                    yield [[episode_train_img[i:i+1]], [network_base_img], [network_base_label]], episode_train_label[i:i+1]
            else:
                #print(episode_test_img.shape[0])
                #assert(0)
                #assert(episode_test_img.shape[0] == 75)
                #assert(self.idx < 50)
                for i in range(test_epoch_size):
                    #print('i',i)
                    yield [[episode_test_img[i:i+1]], [network_base_img], [network_base_label]], episode_test_label[i:i+1]





        
keras_gen_train = KerasBatchGenerator()
gen_train = keras_gen_train.generate()

gen_test = KerasBatchGenerator().generate('test')

print('train data check')
for _ in range(3):
    img, l = next(gen_train)
    print(img.shape,l.shape)
print('test data check')    
for _ in range(3):
    img, l = next(gen_test)
    print(img.shape,l.shape)

import tensorflow as tf
if tf.__version__ < "2.0":
    from tensorflow.keras.backend import set_session
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = args.gpu_mem
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
else:
    #tensorflow 2.0 sets memory growth per default
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Input, Flatten, Conv2D, Lambda, TimeDistributed, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K
import tensorflow.keras

inputs = Input(shape=(None,84,84,3))
print('the shape', inputs.shape)
conv1 = TimeDistributed(Conv2D(50, 7, 1 , activation = 'relu'))(inputs)
conv2 = TimeDistributed(MaxPooling2D(pool_size = (3,3)))(conv1)
conv3 = TimeDistributed(Conv2D(100, 7, 1 , activation = 'relu'))(conv2)
conv4 = TimeDistributed(MaxPooling2D(pool_size = (3,3)))(conv3)
#conv3 = TimeDistributed(Conv2D(5, 5, (3,3) , padding='same', activation = 'relu'))(conv2)
flat = TimeDistributed(Flatten())(conv4)
x = TimeDistributed(Dense(100, activation = 'relu'))(flat)
predictions = Activation('softmax')(x)

model_img = Model(inputs=inputs, outputs=predictions)

#model_img.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])

print(model_img.summary(line_length=180, positions = [.33, .55, .67, 1.]))



input1 = Input(shape=(None,84,84,3))
input2 = Input(shape=(None,84,84,3)) #, tensor = K.variable(episode_train_img[0:0]))

encoded_l = model_img(input1)
encoded_r = model_img(input2)
    
# Add a customized layer to compute the absolute difference between the encodings
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_r])
    
# Add a dense layer with a sigmoid unit to generate the similarity score
prediction = Dense(1)(L1_distance)
    
# Connect the inputs with the outputs
siamese_net = Model(inputs=[input1,input2],outputs=prediction)

#siamese_net.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])
print(siamese_net.summary(line_length=180, positions = [.33, .55, .67, 1.]))


input_lambda1 = Input(shape=(1,84,84,3))
input_lambda2 = Input(shape=(None,84,84,3))
input_lambda3 = Input(shape=(None,cathegories))

s_res = siamese_net([input_lambda1, input_lambda2])

def call(x):
    [k0,l2] = x
    #k0 = siamese_net([x1,x2])
    #k1 = K.expand_dims(tf.reshape(k0, (-1,1)), axis=0)
    k2 = k0 * l2
    r = K.sum(k2, axis = 1)
    print('l2',l2.shape,'k0',k0.shape, 'k2',k2.shape, 'r',r.shape)
    return r
#def call_shape(input_shape):
#    return (5,)

call_lambda = Lambda(call)([s_res, input_lambda3])
call_lambda_softmax = Activation('softmax')(call_lambda)

lambda_model = Model(inputs = [input_lambda1, input_lambda2, input_lambda3], outputs = call_lambda_softmax)

from tensorflow.keras import optimizers as op

if args.pretrained_name is not None:
    from tensorflow.keras.models import load_model
    lambda_model = load_model(args.pretrained_name, custom_objects = { "keras": tensorflow.keras , "args":args})
    print("loaded model",lambda_model)

#after loading to set learning rate
lambda_model.compile(loss='categorical_crossentropy', optimizer=op.SGD(args.lr), metrics=['categorical_accuracy'])
print(lambda_model.summary(line_length=180, positions = [.33, .55, .67, 1.]))


# models in models forget the layer name, therefore one must use the automatically given layer name and iterate throught the models by hand
# here we can try setting the layer not trainable
def all_layers(model):
    layers = []
    for l in model.layers:
        #print(l.name, l.trainable, isinstance(l,Model))
        if isinstance(l, Model):
            a = all_layers(l)
            #print(a)
            layers.extend(a)
        else:
            layers.append(l)
    return layers
       
for l in all_layers(siamese_net):
    l2=l
    if isinstance(l,TimeDistributed):
        l2=l.layer
    print(l2.name,l2.trainable, len(l2.get_weights()))

for l in all_layers(lambda_model):
    l2=l
    p='normal'
    if isinstance(l,TimeDistributed):
        l2=l.layer
        p='timedi'
#    if (l2.name == 'dense_1'):
#        l2.trainable = False
    print(p,l2.name,l2.trainable, len(l2.get_weights()))
    
#lambda_model.get_layer("dense_1").trainable = False
        
# testing with additional batch axis ?!
i=1
test_lambda = lambda_model([K.expand_dims(K.variable(base_train_img[0:0+1]),axis=0),K.expand_dims(K.variable(base_train_img), axis=0), K.expand_dims(K.variable(base_train_label), axis=0)])
#        
print('test lambda', K.eval(test_lambda))



checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)
tensorboard = TensorBoard(log_dir = args.tensorboard_log_dir)
lambda_model.fit_generator(keras_gen_train.generate_add_samples(), train_epoch_size, args.epochs, 
                           validation_data=keras_gen_train.generate_add_samples('test'), validation_steps=test_epoch_size, callbacks = [tensorboard], workers = 0) 
#workers = 0 is a work around to correct the number of calls to the validation_data generator
lambda_model.save(args.final_name+'.hdf5')


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


#weight_grads = get_layer_output_grad(lambda_model, [[episode_train_img[0:1]], [episode_train_img[:]], [episode_train_label[:]]],  [episode_train_label[0:1]])

#weight_grads = get_layer_output_grad(siamese_net, [episode_train_img[0:1],episode_train_img[0:1]],  episode_train_label[0:1])

#print(weight_grads)
#
#input_few = Input(shape=(84,84,3))
#input_labels = Input(shape=(84,84,3))
#
#output_few = Lambda(call)([input_few,K.variable(episode_train_img), K.variable(episode_train_label)])
#
#model_few = Model(inputs = [input_few, input_labels], outputs = output_few)
#
#print('test few', K.eval(model_few([K.variable(episode_train_img[0:0+1]),K.variable(episode_train_label)])))
    
##sum_few = np.zeros(episode_train_label[0:1].shape)
#input_few = Input(shape=(84,84,3))
#for i in range(0,2):
#    a = siamese_net([input_few, K.variable(episode_train_img[i:i+1])])
#    if i == 0:
#        sum_few = K.variable(episode_train_label[i:i+1]) * a
#    else:
#        sum_few += K.variable(episode_train_label[i:i+1]) * a
#    print(i)
#sum_few_softmax = Activation('softmax')(sum_few)
#full_few_shot = Model(inputs = input_few, outputs = sum_few_softmax)
#
##print('net_ready')
##aa = K.variable(episode_train_img[0:1])
##a = full_few_shot(aa)
##print('net ready', K.eval(a))
##
#from tensorflow.keras.optimizers import  Adam
#full_few_shot.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_accuracy'])
#
#print(full_few_shot.summary())
#
#
#print("eval", episode_train_img[0:1])
#sum_a = np.zeros(episode_train_label[0:1].shape)
#for i in range(0,train_epoch_size):
#    a = siamese_net([K.variable(episode_train_img[0:1]), K.variable(episode_train_img[i:i+1])])
#    sum_a += K.variable(episode_train_label[i:i+1]) * a
#    print('aaaaa',i,K.eval(a), episode_train_label[0:1], episode_train_label[i:i+1])
#
#
##print('suma',episode_train_label[0:1], K.eval(K.softmax(sum_a)), K.eval(full_few_shot(K.variable(episode_train_img[0:1]))))
#
#checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)
#
#history = full_few_shot.fit_generator(gen_train.generate(), train_epoch_size, 100, validation_data=gen_test, validation_steps=test_epoch_size) 
#
