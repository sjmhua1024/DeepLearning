# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:45:00 2018

@author: win10
"""

import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#创建环境内物体对象的类
class gameOb():
    #属性：
    #coordinates(x，y坐标)
    #size(尺寸)，
    #intensity(亮度值)
    #channel(RGB通道颜色,0,1,2分别代表红绿蓝)
    #reward(奖励值)
    #name(名称)
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

#创建GridWorld环境的类
class gameEnv():
    def __init__(self,size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        plt.imshow(a,interpolation='nearest')
    
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        goal = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal)
        goal2 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal2)
        goal3 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(goal4)
        hole = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole)
        hole2 = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole2)
        state = self.renderEnv()
        self.state = state
        return state
    
    #实现移动英雄角色的方法
    def moveChar(self,direction):
        hero = self.objects[0]
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        self.objects[0] = hero
    
    #选择一个跟现有物体不冲突的位置
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if(objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]
    
    #检查是否触碰的函数（hero是否触碰了goal或者fire）
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(),1,1,1,1,'goal'))
                else:
                    self.objects.append(gameOb(self.newPosition(),1,1,0,-1,'fire'))
                return other.reward,False
        return 0.0,False
    
    def renderEnv(self):
        a = np.ones([self.sizeY+2,self.sizeX+2,3])
        a[1:-1,1:-1,:] = 0
        hero = None
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,item.x+1:item.x+item.size+1,item.channel] = item.intensity
        b = scipy.misc.imresize(a[:,:,0],[84,84,1],interp='nearest')
        c = scipy.misc.imresize(a[:,:,1],[84,84,1],interp='nearest')
        d = scipy.misc.imresize(a[:,:,2],[84,84,1],interp='nearest')
        a = np.stack([b,c,d],axis=2)
        return a
    
    #执行一步action
    def step(self,action):
        self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderEnv()
        return state,reward,done

#初始化GridWorld环境
env = gameEnv(size=5)


#定义DQN网络
class Qnetwork():
    def __init__(self,h_size):
        #定义DQN的四层卷积神经网络
        self.scalarInput = tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = tf.contrib.layers.conv2d(inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID',biases_initializer=None)
        self.conv2 = tf.contrib.layers.conv2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID',biases_initializer=None)
        self.conv3 = tf.contrib.layers.conv2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID',biases_initializer=None)
        self.conv4 = tf.contrib.layers.conv2d(inputs=self.conv3,num_outputs=512,kernel_size=[7,7],stride=[1,1],padding='VALID',biases_initializer=None)
        
        #采用Dueling DQN思想，将卷积层的输出分为状态价值流(streamVC)和动作价值流(streamVC)
        self.streamAC,self.StreamVC = tf.split(self.conv4,2,3)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([h_size//2,env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,reduction_indices=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout,self.actions_onehot),reduction_indices=1)
        
        #定义loss(targetQ和Q的均方误差)，并使用Adam优化器进行优化
        self.td.error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        
#实现Experience Replay策略，创建一个用来存储Experience（Agent的样本）的缓存buffer
#定义experience_buffer类
class experience_buffer():
    def __init__(self,buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(self.buffer) + len(experience)-self.buffer_size)] = []
        self.buffer.extend(experience)
        
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
 
#将84x84x3的states扁平化为1维向量的函数
def processState(states):
    return np.reshape(states,[21168])       

#更新target DQN模型参数的方法（主DQN使用Qnetwork类中的self.updateModel更新参数）
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau)+((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)