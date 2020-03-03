
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import tensorflow as tf 
from tensorflow.keras.losses import SparseCategoricalCrossentropy
logitloss = SparseCategoricalCrossentropy(from_logits=True)

logits1 = np.zeros((1,3))
logits1[0,:] = [1,2,3]
logits1 = tf.convert_to_tensor(logits1)
actions1 = tf.convert_to_tensor([[1]], dtype = tf.int32)

logits2 = tf.convert_to_tensor(np.random.rand(64,3))
actions2 = tf.convert_to_tensor(np.ones((64,1)), dtype = tf.int32)

#method 1 - using sparse categorical crossentropy 
out1 = logitloss(actions1,logits1)
out2 = logitloss(actions2, logits2)

#method 2 - manually compute what I think sparse categorical crossentropy does
def mylogitsloss(actions, logits):
    p = tf.math.exp(logits)
    p = p /  tf.repeat(tf.math.reduce_sum(p, axis = 1,keepdims = True), 3,1) #probabilities
    a = tf.expand_dims(tf.range(0, len(actions), dtype = tf.int32), 1)
    actions = tf.concat([a,actions], 1)
    out = tf.gather_nd(p, actions)
    return -tf.math.log(out)

out3 = mylogitsloss(actions1, logits1)
out4 = mylogitsloss(actions2,logits2)

#conclusion - sparse categorical crossentropy will take the average of the batch, and if you don't want it to do that you have
#to simply specify the weights to use when taking the average. 

