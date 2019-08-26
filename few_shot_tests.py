##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## This file tests few shot learning
##
## Prototype learning with tensorflow.keras  by D. Schmicker
##
## using  https://github.com/y2l/mini-imagenet-tools
##
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import ast
from mini_imagenet_dataloader import MiniImageNetDataLoader
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Input, Flatten, Conv2D, Lambda, TimeDistributed, MaxPooling2D, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K
import tensorflow.keras
from tensorflow.keras import optimizers as op
import tensorflow as tf
import argparse
import numpy as np
import random, imageio

parser = argparse.ArgumentParser(description='train recurrent net.')
parser.add_argument('--pretrained_name', dest='pretrained_name',  type=str, default=None)
parser.add_argument('--dataset', dest='dataset',  type=str, default='train')
parser.add_argument('--lr', dest='lr',  type=float, default=1e-3)
parser.add_argument('--epochs', dest='epochs',  type=int, default=10)
parser.add_argument('--hidden_size', dest='hidden_size',  type=int, default=64)
parser.add_argument('--final_name', dest='final_name',  type=str, default='final_model')
parser.add_argument('--shuffle_images', dest='shuffle_images', action='store_true')
parser.add_argument('--enable_idx_increase', dest='enable_idx_increase', action='store_true')
parser.add_argument('--use_independent_base', dest='use_independent_base', action='store_true')
parser.add_argument('--train_indep_and_dependent', dest='train_indep_and_dependent', action='store_true')
parser.add_argument('--tensorboard_logdir', dest='tensorboard_logdir',  type=str, default='./logs')
parser.add_argument('--enable_only_layers_of_list', dest='enable_only_layers_of_list',  type=str, default=None)
parser.add_argument('--episode_test_sample_num', dest='episode_test_sample_num',  type=int, default=15)
parser.add_argument('--biaslayer1', dest='biaslayer1', action='store_true')
parser.add_argument('--biaslayer2', dest='biaslayer2', action='store_true')
parser.add_argument('--shots', dest='shots',  type=int, default=5)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--set_model_img_to_weights', dest='set_model_img_to_weights', action='store_true')

args = parser.parse_args()

# uncomment the following to disable CuDNN support
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
###########################################

def debug(what):
    if args.debug:
        return what
    else:
        return ''
def printdeb(*what):
    if args.debug:
        print(*what)

class OurMiniImageNetDataLoader(MiniImageNetDataLoader):
    # adding functions we need
    def idx_to_big(self, phase, idx):
        if phase=='train':
            all_filenames = self.train_filenames
        elif phase=='val':
            all_filenames = self.val_filenames
        elif phase=='test':
            all_filenames = self.test_filenames
        else:
            print('Please select vaild phase')
        one_episode_sample_num = self.num_samples_per_class*self.way_num
        return ((idx+1)*one_episode_sample_num >= len(all_filenames))

    def process_batch(self, input_filename_list, input_label_list, batch_sample_num, reshape_with_one=True, dont_shuffle_batch = False):
        new_path_list = []
        new_label_list = []
        for k in range(batch_sample_num):
            class_idxs = list(range(0, self.way_num))
            if not dont_shuffle_batch:
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

    def get_batch(self, phase='train', idx=0, dont_shuffle_batch = False):
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

        one_episode_sample_num = self.num_samples_per_class*self.way_num
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

        this_inputa, this_labela = self.process_batch(this_task_tr_filenames, this_task_tr_labels, epitr_sample_num, reshape_with_one=False, dont_shuffle_batch = dont_shuffle_batch)
        this_inputb, this_labelb = self.process_batch(this_task_te_filenames, this_task_te_labels, epite_sample_num, reshape_with_one=False, dont_shuffle_batch = dont_shuffle_batch)

        return this_inputa, this_labela, this_inputb, this_labelb


cathegories = 5
shots = args.shots
dataloader = OurMiniImageNetDataLoader(shot_num=shots * 2, way_num=cathegories, episode_test_sample_num=args.episode_test_sample_num, shuffle_images = args.shuffle_images) #twice shot_num is because one might be uses as the base for the samples

dataloader.generate_data_list(phase=args.dataset)

printdeb('mode is',args.dataset)
dataloader.load_list(args.dataset)

base_train_img, base_train_label, base_test_img, base_test_label = \
        dataloader.get_batch(phase=args.dataset, idx=0) 

train_epoch_size = base_train_img.shape[0]
if not args.train_indep_and_dependent:
    train_epoch_size = int(train_epoch_size / 2) # as double is generated for the base and train
test_epoch_size = base_test_img.shape[0]

print("epoch training size:", train_epoch_size, base_train_label.shape[0], "epoch testing size", test_epoch_size)

class KerasBatchGenerator(object):
    # def generate(self, phase='train'):
    #     while True:
    #         if phase == 'train':
    #             for i in range(train_epoch_size):
    #                 yield base_train_img[i:i+1], base_train_label[i:i+1]
    #         else:
    #             for i in range(test_epoch_size):
    #                 yield base_test_img[i:i+1], base_test_label[i:i+1]

    def generate_add_samples(self, phase = 'train'):
        self.idx = 0
        while True:
            batch_train_img, batch_train_label, episode_test_img, episode_test_label = \
                dataloader.get_batch(phase=args.dataset, idx=self.idx, dont_shuffle_batch = (self.idx==0))

            # this depends on what we are trying to train.
            # care must be taken, that with a different dataset the labels have a different meaning. Thus if we use a new dataset, we must 
            # use network_base which fits to the database. Therefore there must be taken images with label from the same dataset.
            network_base_img = batch_train_img[:train_epoch_size]
            network_base_label = batch_train_label[:train_epoch_size]

            #only half is used now, as the rest is reserved for independend base
            episode_train_img = batch_train_img[train_epoch_size:]
            episode_train_label = batch_train_label[train_epoch_size:]
            
            if not args.use_independent_base:
                network_base_img = episode_train_img
                network_base_label = episode_train_label
            if args.train_indep_and_dependent:   #train_epoch_size wrong, before should be old ....
                network_base_img = batch_train_img[:int(train_epoch_size/2)]
                network_base_label = batch_train_label[:int(train_epoch_size/2)]
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
                self.e_t_i = episode_train_img
                self.n_b_i = network_base_img
                self.n_b_l = network_base_label
                self.e_t_l = episode_train_label
                for i in range(train_epoch_size):
                    yield [[episode_train_img[i:i+1]], [network_base_img], [network_base_label]], episode_train_label[i:i+1]
            else:
                for i in range(test_epoch_size):
                    yield [[episode_test_img[i:i+1]], [network_base_img], [network_base_label]], episode_test_label[i:i+1]

keras_gen_train = KerasBatchGenerator()
#gen_train = keras_gen_train.generate()

#gen_test = KerasBatchGenerator().generate('test')

# print('train data check')
# for _ in range(3):
#     img, l = next(gen_train)
#     print(img.shape,l.shape)
# print('test data check')    
# for _ in range(3):
#     img, l = next(gen_test)
#     print(img.shape,l.shape)

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


class FindModel(Model):
    #allows isinstance to find exactly this model as a submodel of other models
    pass

def get_FindModel(model):
    found = None
    for l in model.layers:
        if isinstance(l,FindModel):
            if found is not None:
                raise Exception('two FindModels present')
            else:
                found = l
        if isinstance(l,Model):
            s = get_FindModel(l)
            if s is not None:
                if found is not None:
                    raise Exception('two FindModels present')
                else:
                    found = s
    return found

class BiasLayer(Layer):

    def __init__(self, proto_num, do_bias, bias_num, **kwargs):
        self.proto_num = proto_num
        self.do_bias = do_bias
        self.bias_num = bias_num
        print('mult bias',do_bias, proto_num)
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                      shape=(self.proto_num) + input_shape[2:],
                                      initializer='uniform',
                                      trainable=True)
        if self.do_bias:
            preset = 'ones'
        else:
            preset = 'zeros'
        self.bias_enable = self.add_weight(name='bias_enable',
                                      shape=(1),
                                      initializer=preset,
                                      trainable=False)
        print('bias_enable',self.bias_enable, K.eval(self.bias_enable[0]),'bias',debug(self.bias))
        super(BiasLayer, self).build(input_shape)  # Be sure to call this at the end

    def set_bias(self, do_bias):
        was_weights = self.get_weights()
        if do_bias:
            self.set_weights([was_weights[0],np.array([1])])
            self.trainable = True
        else:
            self.set_weights([was_weights[0],np.array([0])])
            self.trainable = False

    def call(self, x):
        return self.bias * self.bias_enable       + x * (1-self.bias_enable)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'proto_num': self.proto_num, 'do_bias' : self.do_bias,'bias_num' : self.bias_num}

# Network definition starts here

inputs = Input(shape=(None,84,84,3))
printdeb('the shape', inputs.shape)
conv1 = TimeDistributed(Conv2D(args.hidden_size, 3, padding='same', activation = 'relu'))(inputs)
pool1 = TimeDistributed(MaxPooling2D(pool_size = 2))(conv1)
conv2 = TimeDistributed(Conv2D(args.hidden_size, 3, padding='same', activation = 'relu'))(pool1)
pool2 = TimeDistributed(MaxPooling2D(pool_size = 2))(conv2)
conv3 = TimeDistributed(Conv2D(args.hidden_size, 3, padding='same', activation = 'relu'))(pool2)
pool3 = TimeDistributed(MaxPooling2D(pool_size = 2))(conv3)
conv4 = TimeDistributed(Conv2D(args.hidden_size, 3, padding='same', activation = 'relu'))(pool3)
pool4 = TimeDistributed(MaxPooling2D(pool_size = 2))(conv4)
conv5 = TimeDistributed(Conv2D(args.hidden_size, 3, padding='same', activation = 'relu'))(pool4)
pool5 = TimeDistributed(MaxPooling2D(pool_size = 2))(conv5)

flat = TimeDistributed(Flatten())(pool5)
#x = TimeDistributed(Dense(100, activation = 'relu'))(flat)
#predictions = Activation('softmax')(x)

model_img = FindModel(inputs=inputs, outputs=flat)

#model_img.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])

print(model_img.summary(line_length=180, positions = [.33, .55, .67, 1.]))

input1 = Input(shape=(None,84,84,3))
input2 = Input(shape=(None,84,84,3)) #, tensor = K.variable(episode_train_img[0:0]))

input2b = BiasLayer(shots * cathegories, args.biaslayer1, 1)(input2)
encoded_l = model_img(input1)
encoded_r = model_img(input2b)

encoded_rb = BiasLayer(shots * cathegories, args.biaslayer2, 2)(encoded_r)
# Add a customized layer to compute the absolute difference between the encodings
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([encoded_l, encoded_rb])
    
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
    k2 = k0 * l2
    r = K.sum(k2, axis = 1)
    printdeb('l2',l2.shape,'k0',k0.shape, 'k2',k2.shape, 'r',r.shape)
    return r

call_lambda = Lambda(call)([s_res, input_lambda3])
call_lambda_softmax = Activation('softmax')(call_lambda)

lambda_model = Model(inputs = [input_lambda1, input_lambda2, input_lambda3], outputs = call_lambda_softmax)

if args.pretrained_name is not None:
    from tensorflow.keras.models import load_model
    lambda_model = load_model(args.pretrained_name, custom_objects = { "keras": tensorflow.keras , "args":args, "BiasLayer": BiasLayer, "FindModel": FindModel})
    print("loaded model",lambda_model)

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
       
lambda_model_layers = all_layers(lambda_model)
for l in range(len(lambda_model_layers)):
    l2=lambda_model_layers[l]
    p='normal'
    if isinstance(l2,TimeDistributed):
        l2=l2.layer
        p='timedi'
    if args.enable_only_layers_of_list is not None:
        l2.trainable = False
    if isinstance(l2,BiasLayer):
        printdeb('pre ',l2.bias_num, l2.do_bias,args.biaslayer1,args.biaslayer2)
        if (l2.bias_num == 1):
            l2.set_bias(args.biaslayer1)
        if (l2.bias_num == 2):
            l2.set_bias(args.biaslayer2)
            #print('get_weights = ', l2.get_weights())
        printdeb('past',l2.bias_num, l2.do_bias,args.biaslayer1,args.biaslayer2) #, l2.bias)

    print('{:10} {:10} {:20} {:10}  {:10}'.format(l, p,l2.name, ("fixed", "trainable")[l2.trainable], l2.count_params()))

if args.enable_only_layers_of_list is not None:
    print('\nenable some layers for training')

    for i in ast.literal_eval(args.enable_only_layers_of_list):
        lambda_model_layers[i].trainable = True

    for l in range(len(lambda_model_layers)):
        l2=lambda_model_layers[l]
        p='normal'
        if isinstance(l2,TimeDistributed):
            l2=l2.layer
            p='timedi'
        print('{:10} {:10} {:20} {:10}  {:10}'.format(l, p,l2.name, ("fixed", "trainable")[l2.trainable], l2.count_params()))

#after loading to set learning rate
lambda_model.compile(loss='categorical_crossentropy', optimizer=op.SGD(args.lr), metrics=['categorical_accuracy'])
print(lambda_model.summary(line_length=180, positions = [.33, .55, .67, 1.]))

#print('vor fitting', lambda_model_layers[17].get_weights()[0])

checkpointer = ModelCheckpoint(filepath='checkpoints/model-{epoch:02d}.hdf5', verbose=1)
tensorboard = TensorBoard(log_dir = args.tensorboard_logdir)
lambda_model.fit_generator(keras_gen_train.generate_add_samples(), train_epoch_size, args.epochs, 
                           validation_data=keras_gen_train.generate_add_samples('test'), validation_steps=test_epoch_size, callbacks = [tensorboard], workers = 0) 
#workers = 0 is a work around to correct the number of calls to the validation_data generator

#test_lambda = lambda_model([K.expand_dims(K.variable(base_train_img[0:0+1]),axis=0),K.expand_dims(K.variable(base_train_img), axis=0), K.expand_dims(K.variable(base_train_label), axis=0)])
i=0
test_lambda = lambda_model([K.expand_dims(K.variable(keras_gen_train.e_t_i[i:i+1]),axis=0), K.expand_dims(K.variable(keras_gen_train.n_b_i),axis=0),
                            K.expand_dims(K.variable(keras_gen_train.n_b_l),axis=0)], K.expand_dims(K.variable(keras_gen_train.e_t_l[i:i+1]),axis=0))

print(test_lambda)
in_test = lambda_model.input
out_test = lambda_model_layers[22].output
functor = K.function([in_test], [out_test])
print(functor([K.expand_dims(K.variable(keras_gen_train.e_t_i[i:i+1]),axis=0), K.expand_dims(K.variable(keras_gen_train.n_b_i),axis=0),
                            K.expand_dims(K.variable(keras_gen_train.n_b_l),axis=0)]))

find_conv_model = None

def print_FindModels(model):
    found = 0
    for l in model.layers:
        if isinstance(l,FindModel):
            print('FoundModel', l)
            found +=1
        if isinstance(l,Model):
            found += print_FindModels(l)
    return found

#check if allways one
print('number of find models found', print_FindModels(lambda_model))

find_conv_model = get_FindModel(lambda_model)

in_test = find_conv_model.input
out_test = find_conv_model.output
functor = K.function([in_test], [out_test])

calc_out = functor([K.expand_dims(K.variable(keras_gen_train.n_b_i),axis=0)])

printdeb('calc_out',calc_out[0])

for l in lambda_model_layers:
        if isinstance(l,BiasLayer) and l.bias_num == 2:
            printdeb('vor', l.get_weights()[0])

if args.set_model_img_to_weights:
    print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    for l in lambda_model_layers:
        if isinstance(l,BiasLayer) and l.bias_num == 2:
            print("biaslayer2 found",l,l.bias_num)
            l.set_weights([calc_out[0][0],np.array([0])])
            print('nach l', l.get_weights()[0])
    #print('nach [17]', lambda_model_layers[17].get_weights()[0])

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')

for l in range(len(lambda_model_layers)):
    l2=lambda_model_layers[l]
    p='normal'
    if isinstance(l2,TimeDistributed):
        l2=l2.layer
        p='timedi'
    if args.enable_only_layers_of_list is not None:
        l2.trainable = False
    if isinstance(l2,BiasLayer):
        printdeb('pre ',l2.bias_num, l2.do_bias,args.biaslayer1,args.biaslayer2)
        if (l2.bias_num == 1):
            l2.do_bias = l2.trainable = args.biaslayer1
        if (l2.bias_num == 2):
            l2.do_bias = l2.trainable = args.biaslayer2
        printdeb('past',l2.bias_num, l2.do_bias,args.biaslayer1,args.biaslayer2, debug(l2.bias))
    print('{:10} {:10} {:20} {:10}  {:10}'.format(l, p,l2.name, ("fixed", "trainable")[l2.trainable], l2.count_params()), debug(l2.get_weights()))

for l in range(len(lambda_model_layers)):
    lambda_model_layers[l].trainable = True
lambda_model.save(args.final_name+'.hdf5')

# tools for debugging
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
