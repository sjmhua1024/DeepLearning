# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:20:45 2018

@author: win10
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

sess = tf.InteractiveSession()

'''定义算法公式（神经网络forward时的计算）'''
x = tf.placeholder(tf.float32,[None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32,[None,10])

'''定义损失函数loss，选定优化器，指定优化器优化loss'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()


'''迭代地对数据进行训练'''
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch_xs,y_:batch_ys})
    

'''在测试集或者验证机上对准确率进行评测'''
correct_prediction = tf.equal(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print(accuracy.eval({x:mnist.validation.images,y_:mnist.validation.labels}))
