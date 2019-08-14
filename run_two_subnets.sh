#!/bin/bash

# This script trains a rnn net with two subnets. The idea is, that the subnets learn different things, in the case of multiplication by hand one net could learn multiplying two digits, the other adding part results.
# The dense_selector selects the subnet to be used, and the SelectSubnetLayer uses this network.
# The training is done in two stages, first the dense_selector is followed by a Dropout Layer to allow to learn what should be done by which subnet.
# In the second stage the training is done without Dropout.
# The manipulation of the gradient in the SelectSubnetLayer is not used

train_data_num=200
hidden_size=30
python3 learnmultiply_schriftlich_limit_traindata_subnets.py --train_data_num $train_data_num --epoch_size 5000 --hidden_size $hidden_size --check_data_num 10 --selector_pow 1 --epochs 5 --lstm_num 2 --dropout 0.5
python3 learnmultiply_schriftlich_limit_traindata_subnets.py --train_data_num $train_data_num --epoch_size 5000 --hidden_size $hidden_size --check_data_num 10 --selector_pow 1 --epochs 20 --lstm_num 2 --load_weights_name final_model-weights.hdf5  --final_name fine

