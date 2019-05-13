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
