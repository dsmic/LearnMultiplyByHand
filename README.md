# LearnMultiplyByHand
A neural network is teached multiplication by hand

Some previous work was done e.g. "Learning to Execute" (https://arxiv.org/abs/1410.4615). Trying to break the problem down it seems,
that even simple multiplication is not learned with high accuracy.

Therefore we try to teach multiplication by hand to a neural network:
e.g.

    6046588*80647=  
     48372704   
      00000000   
       36279528   
        24186352   
         42326116  
     00111211100   
     487639182436  
 
The neural network is supposed to produce the full multiplication from the first line.


A sample run with two 5 digit long integers for multiplication:
    
    python3 learnmultiply_schriftlich.py --epochs 15 --lr 0.0001
    
    Epoch 15/15
    100000/100000 [==============================] - 1502s 15ms/step - loss: 0.0028 - categorical_accuracy: 0.9996 - val_loss: 0.0019 - val_categorical_accuracy: 0.9997

Added subnets. The idea is, that different subnets do different tasks, e.g. in case of multiplication by hand one does multiplying digits, the other adding.

```
class SelectSubnetLayer(Layer):
    # this layer takes as input a list of identically shaped inputs and outputs the shape of one of thouse inputs without the first entry in the last axis
    # the first entry in the last axis is used as a selector. From all selectors with softmax every input is scaled and added to result in the output
    # The idea is to select one subnet for a given task. Different subnets can do different task and the selector selects, which subnet is used.
    # the layer supports dropout for the selector (before softmax)
```

run

python3 learnmultiply_schriftlich_limit_traindata_subnets.py --train_data_num 5000 --epoch_size 5000 --hidden_size 50 --check_data_num 10 --epochs 50 --lstm_num 2 --lr 0.0001 --start_console --use_full_select_layer

results in (A and B indicating the subnets used):
```
A-0.76  A-0.74  A-0.75  A-0.84  A-0.87  A-0.87  A-0.86  A-0.83  A-0.80  A-0.85  10
A-0.86  A-0.87  A-0.88  A-0.88  A-0.88  A-0.88  A-0.88  B-0.60  A-0.78  A-0.72  20
A-0.50  A-0.73  A-0.72  A-0.50  A-0.72  A-0.72  A-0.50  A-0.73  A-0.72  A-0.50  30
A-0.71  A-0.72  A-0.73  A-0.73  A-0.75  A-0.73  A-0.73  A-0.73  A-0.74  A-0.73  40
A-0.75  A-0.82  A-0.83  A-0.88  A-0.86  A-0.86  A-0.86  A-0.70  A-0.84  A-0.78  50
A-0.51  A-0.73  A-0.73  A-0.50  A-0.73  A-0.73  A-0.52  A-0.74  A-0.75  A-0.79  60
A-0.79  A-0.87  A-0.82  A-0.73  A-0.73  A-0.73  A-0.75  A-0.88  A-0.88  A-0.88  70
A-0.88  A-0.64  A-0.83  A-0.79  A-0.50  A-0.73  A-0.72  A-0.73  A-0.82  A-0.75  80
A-0.52  A-0.73  A-0.73  A-0.52  A-0.70  A-0.72  A-0.74  A-0.74  A-0.81  A-0.75  90
A-0.73  A-0.78  A-0.84  A-0.73  A-0.88  A-0.78  A-0.88  A-0.84  A-0.60  A-0.67  100
A-0.83  A-0.52  A-0.77  A-0.76  A-0.74  A-0.87  A-0.88  A-0.78  A-0.87  A-0.79  110
A-0.74  A-0.84  A-0.87  A-0.87  A-0.87  A-0.88  A-0.83  A-0.80  A-0.87  A-0.88  120
A-0.77  A-0.87  A-0.88  A-0.88  A-0.88  A-0.86  A-0.86  A-0.86  A-0.71  A-0.85  130
A-0.78  A-0.51  A-0.73  A-0.73  A-0.50  A-0.73  A-0.73  A-0.52  A-0.74  A-0.75  140
A-0.79  A-0.81  A-0.88  A-0.84  A-0.73  A-0.73  A-0.73  A-0.75  A-0.88  A-0.88  150
A-0.88  A-0.78  A-0.80  A-0.78  A-0.79  A-0.78  A-0.69  B-0.58  B-0.70  B-0.76  160
B-0.77  B-0.77  B-0.76  B-0.76  B-0.76  B-0.76  B-0.77  B-0.75  B-0.73  B-0.73  170
B-0.73  B-0.57  B-0.52  B-0.51  B-0.51  B-0.57  B-0.66  B-0.84  B-0.85  B-0.84  180
B-0.76  B-0.75  B-0.79  B-0.86  B-0.82  B-0.85  B-0.74  B-0.73  B-0.68  B-0.53  190
B-0.51  B-0.50  B-0.52  B-0.79  B-0.81  B-0.87  B-0.88  B-0.88  B-0.88  B-0.87  200
B-0.82  B-0.87  B-0.88  B-0.88  B-0.84  B-0.78  B-0.68  B-0.53  B-0.51  B-0.50  210
B-0.50  B-0.78  B-0.75  B-0.87  B-0.88  B-0.88  B-0.87  B-0.86  B-0.81  B-0.88  220
B-0.87  B-0.88  B-0.87  B-0.88  B-0.78  B-0.73  B-0.73  B-0.73  B-0.74  B-0.81  230
B-0.87  B-0.88  B-0.88  B-0.88  B-0.88  B-0.87  B-0.88  B-0.88  B-0.88  B-0.86  240
B-0.86  B-0.88  B-0.83  B-0.74  B-0.73  B-0.73  B-0.73  B-0.78  B-0.86  B-0.88  250
B-0.88  B-0.88  B-0.88  B-0.87  B-0.88  B-0.76  B-0.87  B-0.80  B-0.87  B-0.88  260
B-0.88  B-0.82  B-0.79  B-0.77  B-0.80  B-0.76  B-0.83  B-0.87  B-0.88  B-0.88  270
B-0.88  B-0.87  B-0.85  B-0.88  B-0.85  B-0.88  B-0.82  B-0.80  B-0.85  B-0.74  280
B-0.73  B-0.73  B-0.75  B-0.85  B-0.83  B-0.88  B-0.88  B-0.88  B-0.87  B-0.86  290
B-0.79  B-0.88  B-0.77  B-0.86  B-0.88  B-0.78  B-0.69  B-0.71  B-0.72  B-0.73  300
B-0.77  B-0.85  B-0.82  B-0.88  B-0.88  B-0.83  B-0.81  B-0.82  B-0.80  B-0.88  310
B-0.85  B-0.73  B-0.73  B-0.85  B-0.74  B-0.66  B-0.72  B-0.75  B-0.79  B-0.83  320
B-0.76  B-0.87  B-0.87  B-0.79  
correct: 10/10=1.0
```

One can see, that during multiplying mostly subnet A is used, during adding subnet B.



