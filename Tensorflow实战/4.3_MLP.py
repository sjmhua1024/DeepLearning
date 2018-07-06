# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:02:25 2018

@author: win10
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



if __name__ == '__main__':
 
    #读取mnist数据集并创建会话
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)    
    sess = tf.InteractiveSession()
    
    #输入节点数和隐藏层节点数
    in_units = 784
    h1_units = 300
    
    #使用Variable初始化权重和偏置（W1、b1对应隐藏层；W2、b2对应输出层）
    W1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.zeros([h1_units,10]))
    b2 = tf.Variable(tf.zeros([10]))

    ####################################################################################
    #第一步：定义算法公式（神经网络前馈计算）
    ####################################################################################
    x = tf.placeholder(tf.float32,[None,in_units])
    
    #Dropout的比率（保留节点的概率）
    keep_prob = tf.placeholder(tf.float32)
    
    hidden1 = tf.nn.relu(tf.matmul(x,W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop,W2) + b2)

    ####################################################################################
    #第二步：定义损失函数,选择优化器
    ####################################################################################    
    y_= tf.placeholder(tf.float32,[None,10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
    
    ####################################################################################
    #第三步：训练过程
    ####################################################################################     
    tf.global_variables_initializer().run()
    for i in range(3000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        train_step.run(feed_dict = {x:batch_xs,y_:batch_ys,keep_prob:0.75})
        
    ####################################################################################
    #第四步：对模型进行准确率评测
    ####################################################################################     
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #eval函数与run函数功能类似
    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
    
    
    