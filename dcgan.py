import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, fully_connect, lrelu, bn, sigmoid
from utils import draw_plot_gan
from tensorflow.examples.tutorials.mnist import input_data


class DCGAN(object):
    MODEL = 'DCGAN'

    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'DCGAN'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.Z_DIM = 128
        self.IMAGE_SHAPE = [28, 28, 1]

        self.DATASET_PATH = './DATA/MNIST/'

    
    def discriminator(self, inputs, is_training, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            d_hidden1 = conv2d(inputs, 64, [4, 4], strides=[1, 2, 2, 1], name='d_hidden1')
            d_hidden2 = lrelu(bn(conv2d(d_hidden1, 128, [4, 4], strides=[1, 2, 2, 1], name='d_hidden2'), is_training))
            d_flatten = tf.reshape(d_hidden2, [-1, 7*7*128])
            d_hidden3 = lrelu(bn(fully_connect(d_flatten, 1024, name='d_hidden3'), is_training))
            d_hidden4 = fully_connect(d_hidden3, 1, name='d_hidden4')
            d_logits = sigmoid(d_hidden4)
            
            return d_hidden4, d_logits

    
    def generator(self, inputs, is_training, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            g_hidden1 = fully_connect(inputs, 1024, name='g_hidden1')
            g_hidden2 = lrelu(bn(fully_connect(g_hidden1, 7*7*128), is_training))
            g_reshape = tf.reshape(g_hidden2, [-1, 7, 7, 128])
            g_hidden3 = lrelu(bn(conv2d_t(g_reshape, [None, 14, 14, 64], [4, 4], name='g_hidden3'), is_training))
            g_hidden4 = conv2d_t(g_hidden3, [None, 28, 28, 1], [4, 4], name='g_hidden4')
            g_logtis = sigmoid(g_hidden4)

            return g_hidden4, g_logtis

    
    def build_model(self):
        self.input_x = tf.placeholder(tf.float32, [None]+self.IMAGE_SHAPE)
        self.noise_z = tf.placeholder(tf.float32, [None, self.Z_DIM])
        self.is_train = tf.placeholder(tf.bool)

        self.D_real_logits, D_real = self.discriminator(self.input_x, self.is_train)
        self.G_fake_logits, G_fake = self.generator(self.noise_z, self.is_train)
        self.D_fake_logits, D_fake = self.discriminator(self.G_fake_logits, self.is_train, reuse=True)

        D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_real_logits), logits=self.D_real_logits))
        D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_fake_logits), logits=self.D_fake_logits))
        self.D_loss = D_real_loss + D_fake_loss

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_fake_logits), logits=self.D_fake_logits))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE, beta1=0.5).minimize(self.D_loss, var_list=d_vars)
            self.G_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE*5, beta1=0.5).minimize(self.G_loss, var_list=g_vars)

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
                    batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
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

                sample_z = np.random.uniform(-1, 1, size=(self.N_BATCH, self.Z_DIM))
                # fake_images, _ = self.generator(self.noise_z, is_training=False, reuse=True)
                G_epoch_result = sess.run(self.G_fake_logits, feed_dict={self.noise_z:sample_z, self.is_train:False})
                G_epoch_result = np.reshape(G_epoch_result, [-1, 28, 28])
                figure = draw_plot_gan(G_epoch_result)
                save_path = os.path.join(self.OUTPUT_DIR, self.MODEL_NAME+'_'+str(epoch+1).zfill(3)+'.png')
                figure.savefig(save_path, bbox_inches='tight')

                print('EPOCH: {}\t'.format(epoch+1), 'D_LOSS: {:.8}\t'.format(d_total_loss / total_batch), 'G_LOSS: {:.8}'.format(g_total_loss / total_batch))

                self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            
            self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            print('Finish save model')