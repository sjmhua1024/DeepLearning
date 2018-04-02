# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 19:02:42 2018

@author: win10
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#构建Xavier初始化器
def xavier_init(fan_in,fan_out,constant=1):
    low = - constant * np.sqrt(6.0/(fan_in+fan_out))
    high =  constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

