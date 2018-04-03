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

#定义对训练、测试数据进行标准化处理的函数
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test        

#定义一个获取随机block数据的函数：取一个从0到(len(data)-batch_size)之间的随机整数，再以这个随机数作为block的起始位置，然后顺序取到一个batch_size的数据    
def get_random_block_from_data(data,batach_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

#定义一个去噪自编码器的类
class AdditiveGaussianNoiseAutoencoder(object):
    '''
    构造函数，包括以下几个输入：
        n_input：输入变量数；
        n_hidden：隐含层节点数；
        transfer_function：隐含层激活函数，默认为softplus； 
        optimizer：优化器，默认为Adam
        scale：高斯噪声系数，默认为0.1
    '''
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        #初始化参数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        #定义网络结构
        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+self.scale*tf.random_normal([n_input,]),self.weights['w1']),self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        #定义损失函数并优化
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    #定义初始化权重的函数        
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights
    
    #定义partial_fit函数，计算损失cost并执行一步训练（用一个batch数据）。
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost
    
    #定义calc_total_cost函数，只计算损失cost。
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
              
    #定义transform函数，返回自编码器隐含层的输出结果
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    
    #定义generate函数，将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self,hidden=None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    
    #定义reconstruction函数，整体运行一遍复原过程（即总体运行transform和generate）
    def reconstruction(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    
    #定义getWeights函数，获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    #定义getBiases函数，获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])        
    

if __name__ == '__main__':
     mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
     
     X_train, X_test = standard_scale(mnist.train.images,mnist.test.images)
     
     n_samples = int(mnist.train.num_examples)
     training_epochs = 20
     batch_size = 128
     display_step = 1
     
     autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
     
     for epoch in range(training_epochs):
         avg_cost = 0.0
         total_batch = int(n_samples/batch_size)
         for i in range(total_batch):
             batch_xs = get_random_block_from_data(X_train,batch_size)
             cost = autoencoder.partial_fit(batch_xs)
             avg_cost += cost / n_samples * batch_size
        
         if epoch % display_step == 0:
             print("Epoch:",'%04d' % (epoch+1),'cost=',"{:.9f}".format(avg_cost))
            