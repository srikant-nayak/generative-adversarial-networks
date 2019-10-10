# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:09:13 2019

@author: iist
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
'''
#mnist=pd.read_csv('fashion-mnist_test.csv')
TRAIN = ' http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
#!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
   
#!mkdir MNIST FASHION
#! cp*.gz MNIST FASHION/
#from tensorflow.example.tutorials.mnist 
#import input_data
#mnist=input_data.read_data_set("MNIST FASHION/")
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
learning_rate=.0002
batch_size= 128
epochs = 100000

image_dim = 784
gen_hid_dim =256
dis_hid_dim=256
z_noise_dim=100

def xavier_init(shape):
    return(tf.random_normal(shape=shape,stddev=1./tf.sqrt(shape[0]/2.0)))

#define weight and bias dictionary 
weight = {"disc_h":tf.Variable(xavier_init([image_dim,dis_hid_dim])),
         "disc_final":tf.Variable(xavier_init([dis_hid_dim,1])),
         "gen_h":tf.Variable(xavier_init([z_noise_dim,gen_hid_dim])),
         "gen_final":tf.Variable(xavier_init([gen_hid_dim,image_dim]))}

bias= {"disc_h":tf.Variable(xavier_init([dis_hid_dim])),
         "disc_final":tf.Variable(xavier_init([1])),
         "gen_h":tf.Variable(xavier_init([gen_hid_dim])),
         "gen_final":tf.Variable(xavier_init([image_dim]))}


#computational graph
#define discriminative function

def discriminative(x):
    hidden_layer=tf.nn.relu(tf.add(tf.matmul(x,weight["disc_h"]),bias["disc_h"]))
    final_layer=tf.nn.relu(tf.add(tf.matmul(hidden_layer,weight["disc_final"]),bias["disc_final"]))
    disc_output=tf.nn.sigmoid(final_layer)
    return final_layer,disc_output

def generative(x):
    hidden_layer=tf.nn.relu(tf.add(tf.matmul(x,weight["gen_h"]),bias["gen_h"]))
    final_layer=tf.nn.relu(tf.add(tf.matmul(hidden_layer,weight["gen_final"]),bias["gen_final"]))
    gen_output=tf.nn.sigmoid(final_layer)
    return gen_output

# define placeholder for external input
z_input=tf.placeholder(tf.float32 , shape=[None,z_noise_dim] , name='input_noise')
x_input=tf.placeholder(tf.float32 , shape=[None , image_dim] , name='real_input')

#builing the generative network

with tf.name_scope('generative') as scope:
    output_gen=generative(z_input) #g(z)
    
# building the discriminative network

with tf.name_scope('discriminative') as scope:
    real_output1_disc , real_output_disc = discriminative(x_input)  # D(x)
    fake_output1_disc , fake_output_disc = discriminative(output_gen) # D(G(z))
    
#first kind of loss

with tf.name_scope('discriminative_loss') as scope:
    discriminative_loss= -tf.reduce_mean(tf.log(real_output_disc + 0.0001) + tf.log(1.-fake_output_disc +0.0001))
    
with tf.name_scope('generative_loss') as scope:
    generative_loss=-tf.reduce_mean(tf.log(fake_output_disc +.0001))
    

#tensorboard summary

disc_loss_total = tf.summary.scalar('disc_total_loss',discriminative_loss)
gen_loss_total = tf.summary.scalar('gen_total_loss',generative_loss)

#second kind of loss

with tf.name_scope('discriminative_loss') as scope:
    disc_real_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output1_disc , labels=tf.ones_like(real_output1_disc)))
    disc_fake_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output1_disc , labels=tf.zeros_like(fake_output1_disc)))
    discriminative_loss=disc_real_loss + disc_fake_loss
    
with tf.name_scope('generative_loss') as scope:
    generative_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output1_disc , labels=tf.ones_like(fake_output1_disc)))

#tensorboard summary

disc_loss_real_summary=tf.summary.scalar('disc_real_loss',disc_real_loss)
disc_loss_fake_summary=tf.summary.scalar('disc_fake_loss',disc_fake_loss)
disc_loss_summary=tf.summary.scalar('total_disc_loss',discriminative_loss)

disc_loss_total=tf.summary.merge([disc_loss_real_summary,disc_loss_fake_summary,disc_loss_summary])
gen_loss_total=tf.summary.scalar('gen_loss',generative_loss)


#define the variable

generative_var=[weight["gen_h"],weight["gen_final"],bias["gen_h"],bias["gen_final"]]
discriminative_var=[weight["disc_h"],weight["disc_final"],bias["disc_h"],bias["disc_final"]]

#define the optimizer

with tf.name_scope("optimizer_discriminative") as scope:
    discriminative_optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(discriminative_loss , var_list=discriminative_var)

    
with tf.name_scope("optimizer_generative") as scope:
    generative_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(generative_loss , var_list = generative_var)
    
#initialize the variables

init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
writer=tf.summary.FileWriter(".log/",sess.graph)

for epoch in range(epochs):
    x_batch,_=mnist.train.next_batch(batch_size)
    #generate noise to feed the discriminator
    z_noise=np.random.uniform(-1.,1.,size=[batch_size,z_noise_dim])
    _,disc_loss_epoch=sess.run([discriminative_optimizer,discriminative_loss],feed_dict={x_input:x_batch , z_input:z_noise})
    
    _,gen_loss_epoch=sess.run([generative_optimizer,generative_loss],feed_dict={z_input:z_noise})

    #running the discriminative summary
    summary_disc_loss=sess.run(disc_loss_total,feed_dict={x_input:x_batch,z_input:z_noise})
    writer.add_summary(summary_disc_loss,epoch)
    
    #runnning the generative summary
    summary_gen_loss=sess.run(gen_loss_total,feed_dict={z_input:z_noise})
    writer.add_summary(summary_gen_loss,epoch)
    
    if epoch % 2000 == 0:
        print("step:{0},generative_loss:{1},discriminative_loss:{2}".format(epoch,gen_loss_epoch,disc_loss_epoch))
        
        
#testing
# generate image from noise using generative network

n=6

canvas=np.empty((28*n,28*n))
for i in range(n):
    #noise_input
    z_noise=np.random.uniform(-1.,1.,size=[batch_size,z_noise_dim])
    #generate image from noise
    g=sess.run(output_gen,feed_dict={z_input:z_noise})
    #reverse colour for better display
    g = -1 *(g - 1)
    for j in range(n):
        #draw the generated digits
        canvas[i*28:(i+1)*28 , j*28:(j+1)*28]=g[j].reshape([28,28])
        
plt.figure(figsize=(n,n))    
plt.imshow(canvas , origin="uppar",cmap='gray')
plt.show()



