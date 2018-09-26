import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

from tqdm import tqdm       # visualize progress
from tensorflow.examples.tutorials.mnist import input_data
from utils import draw_plot_gan


class GAN(object):
    MODEL = 'GAN'
    
    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'GAN'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.Z_DIM = 128
        self.IMAGE_SHAPE = [28, 28, 1]

        self.DATASET_PATH = './DATA/MNIST/'


    def discriminator(self, inputs, is_training, reuse=False):
        with tf.variable_scope('discirminator', reuse=reuse):
            with tf.variable_scope('d_hidden1', reuse=reuse):
                d_w1 = tf.get_variable('d_w1', shape=[784, 512], initializer=tf.random_normal_initializer(stddev=0.02))
                d_b1 = tf.get_variable('d_b1', shape=[512], initializer=tf.constant_initializer(0.0))

                d_hidden1 = tf.matmul(inputs, d_w1) + d_b1
                d_hidden1 = tf.contrib.layers.batch_norm(d_hidden1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                d_hidden1 = tf.nn.leaky_relu(d_hidden1)

            with tf.variable_scope('d_hidden2', reuse=reuse):
                d_w2 = tf.get_variable('d_w2', shape=[512, 256], initializer=tf.random_normal_initializer(stddev=0.02))
                d_b2 = tf.get_variable('d_b2', shape=[256], initializer=tf.constant_initializer(0.0))

                d_hidden2 = tf.matmul(d_hidden1, d_w2) + d_b2
                d_hidden2 = tf.contrib.layers.batch_norm(d_hidden2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                d_hidden2 = tf.nn.leaky_relu(d_hidden2)

            with tf.variable_scope('d_hidden3', reuse=reuse):
                d_w3 = tf.get_variable('d_w3', shape=[256, 128], initializer=tf.random_normal_initializer(stddev=0.02))
                d_b3 = tf.get_variable('d_b3', shape=[128], initializer=tf.constant_initializer(0.0))

                d_hidden3 = tf.matmul(d_hidden2, d_w3) + d_b3
                d_hidden3 = tf.contrib.layers.batch_norm(d_hidden3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                d_hidden3 = tf.nn.leaky_relu(d_hidden3)

            with tf.variable_scope('d_hidden4', reuse=reuse):
                d_w4 = tf.get_variable('d_w4', shape=[128, 1], initializer=tf.random_normal_initializer(stddev=0.02))
                d_b4 = tf.get_variable('d_b4', shape=[1], initializer=tf.constant_initializer(0.0))

                d_hidden4 = tf.matmul(d_hidden3, d_w4) + d_b4
                d_logits = tf.nn.sigmoid(d_hidden4)

            return d_logits, d_hidden4

    def generator(self, noise, is_training, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('g_hidden1', reuse=reuse):
                g_w1 = tf.get_variable('g_w1', shape=[128, 256], initializer=tf.random_normal_initializer(stddev=0.02))
                g_b1 = tf.get_variable('g_b1', shape=[256], initializer=tf.constant_initializer(0.0))

                g_hidden1 = tf.matmul(noise, g_w1) + g_b1
                g_hidden1 = tf.contrib.layers.batch_norm(g_hidden1, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                g_hidden1 = tf.nn.leaky_relu(g_hidden1)

            with tf.variable_scope('g_hidden2', reuse=reuse):
                g_w2 = tf.get_variable('g_w2', shape=[256, 256], initializer=tf.random_normal_initializer(0.02))
                g_b2 = tf.get_variable('b_w2', shape=[256], initializer=tf.constant_initializer(0.0))

                g_hidden2 = tf.matmul(g_hidden1, g_w2) + g_b2
                g_hidden2 =tf.contrib.layers.batch_norm(g_hidden2, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                g_hidden2 = tf.nn.leaky_relu(g_hidden2)

            with tf.variable_scope('g_hidden3', reuse=reuse):
                g_w3 = tf.get_variable('g_w3', shape=[256, 512], initializer=tf.random_normal_initializer(stddev=0.02))
                g_b3 = tf.get_variable('g_b3', shape=[512], initializer=tf.constant_initializer(0.0))

                g_hidden3 = tf.matmul(g_hidden2, g_w3) + g_b3
                g_hidden3 = tf.contrib.layers.batch_norm(g_hidden3, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                g_hidden3 = tf.nn.leaky_relu(g_hidden3)

            with tf.variable_scope('g_hidden_conv2', reuse=reuse):
                g_w4 = tf.get_variable('g_w4', shape=[512, 784], initializer=tf.random_normal_initializer(stddev=0.02))
                g_b4 = tf.get_variable('g_b4', shape=[784], initializer=tf.constant_initializer(0.0))

                g_hidden4 = tf.matmul(g_hidden3, g_w4) + g_b4
                g_logits = tf.nn.sigmoid(g_hidden4)

            return g_logits, g_hidden4

    def build_model(self):
        self.input_x = tf.placeholder(tf.float32, [None, self.IMAGE_SHAPE[0]*self.IMAGE_SHAPE[1]])
        self.noise_z = tf.placeholder(tf.float32, [None, self.Z_DIM])
        self.is_train = tf.placeholder(tf.bool)


        self.D_real_logits, D_real = self.discriminator(self.input_x, self.is_train)
        self.G_fake_logits, G_fake = self.generator(self.noise_z, self.is_train)
        self.D_fake_logits, D_fake = self.discriminator(self.G_fake_logits, self.is_train, reuse=True)

        self.D_loss = tf.reduce_mean(tf.log(self.D_real_logits) + tf.log(1 - self.D_fake_logits))
        self.G_loss = tf.reduce_mean(tf.log(self.D_fake_logits))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE, beta1=0.5).minimize(-self.D_loss, var_list=d_vars)
            self.G_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE*5, beta1=0.5).minimize(-self.G_loss, var_list=g_vars)
        
        self.d_loss_sum = tf.summary.merge([tf.summary.scalar("d_loss", self.D_loss)])
        self.g_loss_sum = tf.summary.merge([tf.summary.scalar("g_loss", self.G_loss)])

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)


    def train_model(self):
        if not os.path.exists(self.MODEL_NAME+'_result'):   os.mkdir(self.MODEL_NAME+'_result')
        if not os.path.exists(self.LOGS_DIR):   os.mkdir(self.LOGS_DIR)
        if not os.path.exists(self.CKPT_DIR):   os.mkdir(self.CKPT_DIR)
        if not os.path.exists(self.OUTPUT_DIR): os.mkdir(self.OUTPUT_DIR)

        mnist = input_data.read_data_sets(self.DATASET_PATH, one_hot=True)
        ckpt_save_path = os.path.join(self.CKPT_DIR, self.MODEL_NAME+'_'+str(self.N_BATCH)+'_'+str(self.LEARNING_RATE))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_batch = int(mnist.train.num_examples / self.N_BATCH)
            counter = 0

            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.LOGS_DIR, sess.graph)

            for epoch in tqdm(range(self.N_EPOCH)):
                d_total_loss, g_total_loss = 0, 0

                for i in range(total_batch):
                    batch_xs, _ = mnist.train.next_batch(self.N_BATCH)
                    noise = np.random.uniform(-1, 1, size=(self.N_BATCH, self.Z_DIM))

                    _, d_summary, d_loss = sess.run([self.D_optimizer, self.d_loss_sum, self.D_loss],
                                                    feed_dict={self.input_x:batch_xs, self.noise_z:noise, self.is_train:True})
                    _, g_summary, g_loss = sess.run([self.G_optimizer, self.g_loss_sum, self.G_loss],
                                                    feed_dict={self.noise_z:noise, self.is_train:True})

                    self.writer.add_summary(d_summary, global_step=counter)
                    self.writer.add_summary(g_summary, global_step=counter)
                    counter += 1

                    d_total_loss += d_loss
                    g_total_loss += g_loss

                sample_z = np.random.uniform(-1, 1, size=(5, self.Z_DIM))

                G_epoch_result = sess.run(self.G_fake_logits, feed_dict={self.noise_z:sample_z, self.is_train:False})
                G_epoch_result = np.reshape(G_epoch_result, [-1, 28, 28])
                save_path = os.path.join(self.OUTPUT_DIR, self.MODEL_NAME+'_'+str(epoch+1).zfill(3)+'.png')
                draw_plot_gan(G_epoch_result, save_path)

                print('EPOCH: {}\t'.format(epoch+1), 'D_LOSS: {:.8}\t'.format(d_total_loss / total_batch), 'G_LOSS: {:.8}'.format(g_total_loss / total_batch))

                self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            
            self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            print('Finish save model')

            