# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:51:16 2018

@author: win10
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim

#定义匿名函数trunc_normal，产生截断的正态分布
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0,stddev)

#定义函数inception_v3_arg_scope,用于生成网络中经常用到的函数的默认参数
def inception_v3_arg_scope(weight_decay=0.00004,stddev=0.1,batch_norm_var_collection='moving_vars'):
    #定义归一化参数的字典
    batch_norm_params={
            'decay':0.9997,    #指数衰减函数的更新速度
            'epsilon':0.001,   #归一化时，除以方差，为了防止方差为0而加上的一个数
            'updates_collections':tf.GraphKeys.UPDATE_OPS,  #在当前训练完成后更新均值和方差
            'variables_collections':{
                    'beta':None,
                    'gamma':None,
                    'moving_mean':[batch_norm_var_collection],  #每个批次的均值
                    'moving_variance':[batch_norm_var_collection]   #每个批次的方差                    
                    }
            }

    #slim.arg_scope可以定义一些函数的默认参数值
    #tips:利用list同时定义多个函数的默认参数；允许互相嵌套
    with slim.arg_scope(
            [slim.conv2d,slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay)): #weight_regularizer权重的正则化器
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,  #normalizer_fn用于指定归一化函数
                normalizer_params=batch_norm_params  #normalizer_params用于指定归一化参数,
                ) as sc:
            return sc

#定义函数inception_v3_base,用于生成Inception V3网络中的卷积部分
#参数inputs为输入图片数据的tensor，scope为包含了函数默认参数的环境
def inception_v3_base(inputs,scope=None):
    end_points={}        #用于保存某些关键节点
    
    with tf.variable_scope(scope,'InceptionV3',[inputs]):        
        #定义前七层网络结构(5个卷积和2个池化层交替)
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='VALID'):
            net = slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv2d_1a_3x3')#输入为299x299x3,输出为149x149x32
            net = slim.conv2d(net,32,[3,3],scope='Conv2d_2a_3x3')#输入为149x149x32,输出为147x147x32
            net = slim.conv2d(net,64,[3,3],padding='SAME',scope='Conv2d_2b_3x3')#输入为147x147x32,输出为147x147x64
            net = slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_3a_3x3')#输入为147x147x64,输出为73x73x64
            net = slim.conv2d(net,80,[1,1],scope='Conv2d_3b_1x1')#输入为73x73x64,输出为73x73x80
            net = slim.conv2d(net,192,[3,3],scope='Conv2d_4a_3x3')#输入为73x73x80,输出为71x71x192
            net = slim.max_pool2d(net,[3,3],padding='SAME',scope='MaxPool_5a_3x3')#输入为71x71x192,输出为35x35x192
            
            #定义第1个Inception模块组，包含3个结构类似的Inception Module
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
                #第1个Inception Module
                with tf.variable_scope('Mixed_5b'): #输入为35x35x192
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x64
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x48
                        branch_1 = slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')#输出为35x35x64
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x64
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')#输出为35x35x96
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')#输出为35x35x96
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为35x35x192
                        branch_3 = slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')#输出为35x35x32
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为35x35x256
                #第2个Inception Module
                with tf.variable_scope('Mixed_5c'): #输入为35x35x256
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x64
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x48
                        branch_1 = slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')#输出为35x35x64
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x64
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')#输出为35x35x96
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')#输出为35x35x96
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为35x35x192
                        branch_3 = slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')#输出为35x35x64
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为35x35x288
                #第3个Inception Module
                with tf.variable_scope('Mixed_5d'): #输入为35x35x256
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x64
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x48
                        branch_1 = slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')#输出为35x35x64
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x64
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')#输出为35x35x96
                        branch_2 = slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')#输出为35x35x96
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为35x35x192
                        branch_3 = slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')#输出为35x35x64
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为35x35x288               
           
            #定义第2个Inception模块组，包含5个Inception Module
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
                #第1个Inception Module
                with tf.variable_scope('Mixed_6a'): #输入为35x35x192
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,384,[3,3],stride=2,padding='VALID',scope='Conv2d_1a_1x1')#输出为17x17x384
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')#输出为35x35x64
                        branch_1 = slim.conv2d(branch_1,96,[3,3],scope='Conv2d_0b_3x3')#输出为35x35x96
                        branch_1 = slim.conv2d(branch_1,96,[3,3],stride=2,padding='VALID',scope='Conv2d_1a_1x1')#输出为17x17x96
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='MaxPool_1a_3x3')#输出为17x17x288
                    net = tf.concat([branch_0,branch_1,branch_2],3)#从输出通道维度对三个分支进行合并，输出为17x17x768
                #第2个Inception Module
                with tf.variable_scope('Mixed_6b'): #输入为17x17x768
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x128
                        branch_1 = slim.conv2d(branch_1,128,[1,7],scope='Conv2d_0b_1x7')#输出为17x17x128
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')#输出为17x17x192                        
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,128,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x128
                        branch_2 = slim.conv2d(branch_2,128,[7,1],scope='Conv2d_0b_7x1')#输出为17x17x128
                        branch_2 = slim.conv2d(branch_2,128,[1,7],scope='Conv2d_0c_1x7')#输出为17x17x128
                        branch_2 = slim.conv2d(branch_2,128,[7,1],scope='Conv2d_0d_7x1')#输出为17x17x128
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0e_1x7')#输出为17x17x192 
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为17x17x768
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')#输出为17x17x192
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为17x17x768
                #第3个Inception Module
                with tf.variable_scope('Mixed_6c'): #输入为17x17x768
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,160,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x160
                        branch_1 = slim.conv2d(branch_1,160,[1,7],scope='Conv2d_0b_1x7')#输出为17x17x160
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')#输出为17x17x192                        
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,160,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0b_7x1')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,160,[1,7],scope='Conv2d_0c_1x7')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0d_7x1')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0e_1x7')#输出为17x17x192 
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为17x17x768
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')#输出为17x17x192
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为17x17x768
                #第4个Inception Module
                with tf.variable_scope('Mixed_6d'): #输入为17x17x768
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,160,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x160
                        branch_1 = slim.conv2d(branch_1,160,[1,7],scope='Conv2d_0b_1x7')#输出为17x17x160
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')#输出为17x17x192                        
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,160,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0b_7x1')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,160,[1,7],scope='Conv2d_0c_1x7')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,160,[7,1],scope='Conv2d_0d_7x1')#输出为17x17x160
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0e_1x7')#输出为17x17x192 
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为17x17x768
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')#输出为17x17x192
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为17x17x768
                #第5个Inception Module
                with tf.variable_scope('Mixed_6e'): #输入为17x17x768
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                        branch_1 = slim.conv2d(branch_1,192,[1,7],scope='Conv2d_0b_1x7')#输出为17x17x192
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')#输出为17x17x192                        
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                        branch_2 = slim.conv2d(branch_2,192,[7,1],scope='Conv2d_0b_7x1')#输出为17x17x192
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0c_1x7')#输出为17x17x192
                        branch_2 = slim.conv2d(branch_2,192,[7,1],scope='Conv2d_0d_7x1')#输出为17x17x192
                        branch_2 = slim.conv2d(branch_2,192,[1,7],scope='Conv2d_0e_1x7')#输出为17x17x192 
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为17x17x768
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')#输出为17x17x192
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为17x17x768
                end_points['Mixed_6e'] = net

            #定义第3个Inception模块组，包含3个Inception Module
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
                #第1个Inception Module
                with tf.variable_scope('Mixed_7a'): #输入为17x17x768
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                        branch_0 = slim.conv2d(branch_0,320,[3,3],stride=2,padding='VALID',scope='Conv2d_1a_3x3')#输出为8x8x320
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,192,[1,1],scope='Conv2d_0a_1x1')#输出为17x17x192
                        branch_1 = slim.conv2d(branch_1,192,[1,7],scope='Conv2d_0b_1x7')#输出为17x17x192
                        branch_1 = slim.conv2d(branch_1,192,[7,1],scope='Conv2d_0c_7x1')#输出为17x17x192
                        branch_1 = slim.conv2d(branch_1,192,[3,3],stride=2,padding='VALID',scope='Conv2d_1a_3x3')#输出为8x8x192                        
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net,[3,3],stride=2,padding='VALID',scope='MaxPool_1a_3x3')#输出为8x8x768
                    net = tf.concat([branch_0,branch_1,branch_2],3)#从输出通道维度对三个分支进行合并，输出为8x8x1280
                #第2个Inception Module
                with tf.variable_scope('Mixed_7b'): #输入为8x8x1280
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')#输出为8x8x320
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')#输出为8x8x384
                        branch_1 = tf.concat([
                                slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),
                                slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0b_3x1')],3)#输出为8x8x768(384+384)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')#输出为8x8x448
                        branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')#输出为8x8x384
                        branch_2 = tf.concat([
                                slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),
                                slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')],3)#输出为8x8x768(384+384)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为8x8x1280
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')#输出为8x8x192
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为8x8x2048
                #第3个Inception Module
                with tf.variable_scope('Mixed_7c'): #输入为8x8x1280
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')#输出为8x8x320
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')#输出为8x8x384
                        branch_1 = tf.concat([
                                slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),
                                slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0b_3x1')],3)#输出为8x8x768(384+384)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')#输出为8x8x448
                        branch_2 = slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')#输出为8x8x384
                        branch_2 = tf.concat([
                                slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),
                                slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')],3)#输出为8x8x768(384+384)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')#输出为8x8x1280
                        branch_3 = slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')#输出为8x8x192
                    net = tf.concat([branch_0,branch_1,branch_2,branch_3],3)#从输出通道维度对四个分支进行合并，输出为8x8x2048
                return net,end_points

#定义函数inception_v3，用于生成Inception V3网络最后一部分 —— 全局平均池化、Softmax和Auxiliary Logits
#参数解释：
#num_classes：最后需要分类的数量（默认1000是ILSVRC比赛数据集的种类数）
#is_training：标志是否是训练过程（对Batch Normalization和Dropout有影响）
#dropout_keep_prob：训练时Dropout所需保留节点的比例，默认0.8
#predicion_fn：最后用来进行分类的函数，默认slim.softmax
#spatial_squeeze：标志是否会对输出进行squeeze操作（去除维数为1的维度）
#reuse：标志是否会对网络和Variable进行重复使用
#scope：包含了函数默认参数的环境
def inception_v3(inputs,num_classes=1000,is_training=True,dropout_keep_prob=0.8,prediction_fn=slim.softmax,spatial_squeeze=True,reuse=None,scope='InceptionV3'):
    #定义网络的name、reuse等参数的默认值
    with tf.variable_scope(scope,'InceptionV3',[inputs,num_classes],reuse=reuse) as scope:
        #定义Batch Normalization和Dropout的is_training标志的默认值
        with slim.arg_scope([slim.batch_norm,slim.dropout],is_training=is_training):
            #使用inception_v3_base构筑网络的卷积部分，得到最后一层的输出net，和重要节点的字典表end_points
            net,end_points = inception_v3_base(inputs,scope=scope)
            
            #定义辅助分类Auxiliary Logits的逻辑，使用Mixed_6e的输出
            with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    #输入为17x17x768（Mixed_6e的输出），经过5x5的平均池化，输出为5x5x768
                    aux_logits = slim.avg_pool2d(aux_logits,[5,5],stride=3,padding='VALID',scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits,128,[1,1],scope='Conv2d_1b_1x1') #输出为5x5x128
                    aux_logits = slim.conv2d(
                            aux_logits,768,[5,5],
                            weights_initializer=trunc_normal(0.01),
                            padding='VALID',scope='Conv2d_2a_5x5') #输出为1x1x768
                    aux_logits = slim.conv2d(
                            aux_logits,num_classes,[1,1],activation_fn=None,
                            normalizer_fn=None,weights_initializer=trunc_normal(0.001),
                            scope='Conv2d_2b_1x1') #输出为1x1x1000
                    #消除前两个为1的维度，将输出的维度变为1000
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits,[1,2],name='SpatialSqueeze') #输出维度变为1000
                    end_points['AuxLogits'] = aux_logits
                
                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net,[8,8],padding='VALID',scope='AvgPool_1a_8x8')#输出为1x1x2048
                    net = slim.dropout(net,keep_prob=dropout_keep_prob,scope='Dropout_1b')
                    end_points['PreLogits'] = net
                    logits = slim.conv2d(net,num_classes,[1,1],activation_fn=None,normalizer_fn=None,scope='Conv2d_1c_1x1') #输出为1x1x1000
                    if spatial_squeeze:
                        logits = tf.squeeze(logits,[1,2],name='SpatialSqueeze') #输出维度为1000
                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits,scope='Predictions')
    return logits,end_points
    
    
    
    