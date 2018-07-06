# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:25:53 2018

@author: win10
"""

import tensorflow as tf
import numpy as np
import time
import math
from tensorflow.models.tutorials.image.cifar10 import cifar10,cifar10_input



if __name__ == '__main__':

    #定义常用的常量
    max_steps = 3000
    batch_size = 128
    data_dir = 'data/cifar10_data/cifar-10-batches-bin'
    
    #定义初始化weight的函数
    def variable_with_weight_loss(shape,stddev,w1):
        var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
        if w1 is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')  #需要注意tf.matmul和tf.multiply的区别
            tf.add_to_collection('losses',weight_loss)
        return var
    
    #定义初始化bias的函数
    def bias_variable(shape,con):
        initial = tf.constant(con,shape=shape)
        return tf.Variable(initial)

    #定义计算loss的函数，将softmax和cross entropy loss结合在一起
    def loss(logits,labels):
        labels = tf.cast(labels,tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
        tf.add_to_collection('losses',cross_entropy_mean)
        
        return tf.add_n(tf.get_collection('losses'),name='total_loss')        
    
    #通过cifar10下载数据集
    #通过cifar10_input中的distorted_inputs和inputs生成训练数据和测试数据
    #cifar10.maybe_download_and_extract()
    images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
    images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)    
    
    image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
    label_holder = tf.placeholder(tf.int32,[batch_size])
    
    #创建第一个卷积层
    #卷积核为5*5，输入3通道，输出64通道，步长为1
    #池化层采用最大池化，大小3*3，步长为2
    weight1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=0.05,w1=0.0)
    bias1 = bias_variable([64],0.0)
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME'),bias1))
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
    
    #创建第二个卷积层
    #卷积核为5*5，输入3通道，输出64通道，步长为1
    #池化层采用最大池化，大小3*3，步长为2
    weight2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=0.05,w1=0.0)
    bias2 = bias_variable([64],0.1)
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME'),bias2))
    norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    
    #创建第一个全连接层
    reshape = tf.reshape(pool2,[batch_size,-1])
    dim = reshape.get_shape()[1].value
    weight3 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
    bias3 = bias_variable([384],0.1)
    local3 = tf.nn.relu(tf.matmul(reshape,weight3) + bias3)
    
    #创建第二个全连接层
    weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
    bias4 = bias_variable([192],0.1)
    local4 = tf.nn.relu(tf.matmul(local3,weight4) + bias4)
    
    #输出层
    weight5 = variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
    bias5 = bias_variable([10],0.0)
    logits = tf.add(tf.matmul(local4,weight5),bias5)
    
    #求最终loss（利用loss函数），设置优化函数
    loss = loss(logits,label_holder)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
    
    #输出分数最高那一类的准确率
    top_k_op = tf.nn.in_top_k(logits,label_holder,1)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    #启动图片数据增强的线程队列
    tf.train.start_queue_runners()
    
    #训练过程
    for step in range(max_steps):
        start_time =time.time()
        image_batch,label_batch = sess.run([images_train,labels_train])
        _,loss_value = sess.run([train_op,loss] , feed_dict={image_holder:image_batch,label_holder:label_batch})
        duration = time.time() - start_time
        
        if step % 10 == 0:
            examples_per_sec = batch_size/duration
            sec_per_batch = float(duration)
            
            format_str=('step %d,loss=%.2f (%.1f examples/sec ; %.3f sec/batch)')
            print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))
            
    
    #评估测试集上的准确率
    num_examples = 10000            
    num_iter = int(math.ceil(num_examples/batch_size))
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        image_batch,label_batch = sess.run([images_test,labels_test])
        predictions = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
        true_count += np.sum(predictions)
        step += 1
    
    precision = true_count/total_sample_count
    print('precision @ 1 = %.3f' % precision)
    
    
    
    
    
    