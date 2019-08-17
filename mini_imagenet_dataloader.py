##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@nus.edu.sg
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import random
import numpy as np
from tqdm import trange
import imageio

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

        self.npy_base_dir = npy_dir + str(self.shot_num) + 'shot_' + str(self.way_num) + 'way_' + str(episode_test_sample_num) + '/'
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
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num), nb_samples=self.num_samples_per_class, shuffle=False)
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
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num), nb_samples=self.num_samples_per_class, shuffle=False)
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
                    labels_and_images = self.get_images(sampled_character_folders, range(self.way_num), nb_samples=self.num_samples_per_class, shuffle=False)
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

        this_inputa, this_labela = self.process_batch(this_task_tr_filenames, this_task_tr_labels, epitr_sample_num, reshape_with_one=False)
        this_inputb, this_labelb = self.process_batch(this_task_te_filenames, this_task_te_labels, epite_sample_num, reshape_with_one=False)

        return this_inputa, this_labela, this_inputb, this_labelb

cathegories = 5
dataloader = MiniImageNetDataLoader(shot_num=5, way_num=cathegories, episode_test_sample_num=15)

dataloader.generate_data_list(phase='train', episode_num = 20000)
#dataloader.generate_data_list(phase='val')
#dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='train')

episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
        dataloader.get_batch(phase='train', idx=0)

train_epoch_size = episode_train_img.shape[0]
test_epoch_size = episode_test_img.shape[0]

print("epoch training size:", train_epoch_size, episode_train_label.shape[0], "epoch testing size", test_epoch_size)

class KerasBatchGenerator(object):

    def __init__(self, phase = 'train'):
        self.phase = phase
            
    def generate(self):
        idx = 0
        while True:
            episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
                dataloader.get_batch(phase='train', idx=idx)
            if self.phase == 'train':
                #print(episode_train_img.shape[0])
                for i in range(episode_train_img.shape[0]):
                    yield episode_train_img[i:i+1], episode_train_label[i:i+1]
            else:
                #print(episode_test_img.shape[0])
                for i in range(episode_test_img.shape[0]):
                    yield episode_test_img[i:i+1], episode_test_label[i:i+1]

gen_train = KerasBatchGenerator().generate()
gen_test = KerasBatchGenerator(phase = 'test').generate()

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
from tensorflow.keras.layers import Activation, Dense, Input, Flatten, Conv2D, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

inputs = Input(shape=(84,84,3))
conv = Conv2D(10,3,(3,3))(inputs)
flat = Flatten()(conv)
x = Dense(cathegories)(flat)
predictions = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])

print(model.summary(line_length=180, positions = [.33, .55, .67, 1.]))



input1 = Input(shape=(84,84,3))
input2 = Input(shape=(84,84,3)) #, tensor = K.variable(episode_train_img[0:0]))

encoded_l = model(input1)
encoded_r = model(input2)

# Add a customized layer to compute the absolute difference between the encodings
L1_layer = Lambda(lambda tensors:-10*K.sum(K.square(tensors[0] - tensors[1]), keepdims=True))
L1_distance = L1_layer([encoded_l, encoded_r])

# Add a dense layer with a sigmoid unit to generate the similarity score
#prediction = Dense(1,activation='exponential')(L1_distance)
prediction = Activation('exponential')(L1_distance)

# Connect the inputs with the outputs
siamese_net = Model(inputs=[input1,input2],outputs=prediction)

print("eval", episode_train_img[0:1])
for i in range(10):
    a = siamese_net([K.variable(episode_train_img[0:1]), K.variable(episode_train_img[i:i+1])])
    print('aaaaa',i,K.eval(a), episode_train_label[0:1], episode_train_label[i:i+1])
siamese_net.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['binary_accuracy'])

print(siamese_net.summary())
checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)

history = model.fit_generator(gen_train, train_epoch_size, 10, validation_data=gen_test, validation_steps=test_epoch_size, callbacks=[checkpointer]) 

