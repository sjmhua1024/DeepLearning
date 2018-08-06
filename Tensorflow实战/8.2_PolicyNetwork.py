# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:08:44 2018

@author: win10
"""

import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')


#定义策略网络（带有一个隐含层的MLP）
#设置网络超参数
H = 50                      #隐含层节点数为50
batch_size = 25             
learning_rate = 1e-1        #学习率0.1
D = 4                       #环境信息的维度D为4
gamma = 0.99                #Reward的衰减率为0.99

#定义策略网络的具体结构
observations = tf.placeholder(tf.float32,[None,D],name='input_x')
W1 = tf.get_variable('W1',shape=[D,H],initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable('W2',shape=[H,1],initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)

#定义策略网络优化器
input_y = tf.placeholder(tf.float32,[None,1],name='input_y')        #人工设置的虚拟label
advantages = tf.placeholder(tf.float32,name='reward_signal')        #每个行为的潜在价值
loglik = tf.log(input_y*(input_y-probability)+(1-input_y)*(input_y+probability)) #当前行为对应概率的对数
loss = -tf.reduce_mean(loglik*advantages)       #将行为对应的概率对数与潜在价值相乘，取负数作为损失

tvars = tf.trainable_variables()            #tvars为W1和W2
newGrads = tf.gradients(loss,tvars)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32,name='batch_grad1')
W2Grad = tf.placeholder(tf.float32,name='batch_grad2')
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

#定义估算长期价值（潜在价值）的函数
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for i in reversed(range(r.size)):
        running_add = running_add * gamma + r[i]
        discounted_r[i] = running_add    
    return discounted_r

#定义常用参数
xs,ys,drs = [],[],[]            #xs为环境信息observation的列表，ys为人工设置的虚拟label，drs为记录每一个Action的Reward
reward_sum = 0                  #累计的Reward
episode_number = 1              #
total_episodes = 10000          #总试验次数

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    
    observation = env.reset()
    
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    while episode_number <= total_episodes:
        if reward_sum/batch_size > 100 or rendering == True:
            env.render()
            rendering = True
        x = np.reshape(observation,[1,D])
        
        tfprob = sess.run(probability,feed_dict={observations:x})
        action = 1 if np.random.uniform() < tfprob else 0
        
        xs.append(x)
        y = 1 - action
        ys.append(y)
        
        observation,reward,done,info = env.step(action)
        reward_sum += reward
        drs.append(reward)
        
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs,ys,drs = [],[],[]
            
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
    
            tGrad = sess.run(newGrads,feed_dict={observations:epx,input_y:epy,advantages:discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad            
            
            if episode_number % batch_size == 0:
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                print('Average reward for episode %d : %f.' % (episode_number,reward_sum/batch_size))
                if reward_sum/batch_size >= 200:
                    print('Task solve in',episode_number,'episodes!')
                    break
                reward_sum = 0
            observation = env.reset()
            
                
            