# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:38:17 2018

@author: win10
"""

import collections
import tensorflow as tf

slim = tf.contrib.slim

#利用collections.nametuple设计ResNet基本Block模块组的name tuple，并用它创建Block的类
#需要三个参数：
#scope：Block的名称（或scope）
#unit_fn：构建ResNet的一个残差学习单元（例如bottleneck）
#args：Block的参数信息，形如[(256,64,1)]x2+[(256,64,2)]的列表
##列表中的元素为三元tuple，即(depth,depth_bottleneck,stride):
###depth是第三层输出通道数，depth_bottlenec是前两层输出通道数，stride是步长
class Block(collections.nametuple('Block',['scope','unit_fn','args'])):
    '''A named tuple describing a ResNet block'''
    
#定义下采样函数，参数包括：inputs（输入）、factor（采样因子）、scope
def subsample(inputs,factor,scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs,[1,1],stride=factor,scope=scope)
    
#定义conv2d_same函数，用于创建卷积层
#由于需要保持卷积前后尺寸完全一致，对于步长不为1的卷积，需要进行手动补零操作
def conv2d_same(inputs,num_outputs,kernel_size,stride,scope=None):
    if stride == 1:
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=1,padding='SAME',scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total//2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        return slim.conv2d(inputs,num_outputs,kernel_size,stride=stride,padding='VALID',scope=scope)

#定义堆叠Blocks的函数
#参数中的net为输入，blocks是之前定义的Block的class的列表，outputs_collections是用来收集各个end_points的collections
@slim.add_arg_scope
def stack_blocks_dense(net,blocks,outputs_collections=None):    
    for block in blocks:
        with tf.variable_scope(block.scope,'block',[net]) as sc:
            for i,unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1),values=[net]):
                    unit_depth,unit_depth_bottleneck,unit_stride = unit
                    net = block.unit_fn(net,depth=unit_depth,depth_bottleneck=unit_depth_bottleneck,stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections,sc.name,net)
    return net



        