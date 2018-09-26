import tensorflow as tf
import numpy as np
import os, random
import tensorflow.contrib.slim as slim
import scipy.ndimage.interpolation
import matplotlib.pyplot as plt

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, fully_connect, max_pool, lrelu, bn, sigmoid
from utils import draw_plot_gan
from tensorflow.examples.tutorials.mnist import input_data


class DiscoGAN(object):
    MODEL = 'DiscoGAN'

    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'DiscoGAN'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.BETA1 = 0.5
        self.IMAGE_SHAPE = [28, 28, 1]

        self.DATASET_PATH = './DATA/MNIST/'


    def generator_AB(self, inputs, is_training, reuse=False):
        with tf.variable_scope('Generator_AB', reuse=reuse):
            with tf.variable_scope('g_ab_hidden1'):
                layer1_1 = lrelu(bn(conv2d(inputs, 64, [3, 3], initializer='random', name='conv_1'), is_training))

            with tf.variable_scope('g_ab_hidden2'):
                layer2_1 = max_pool(layer1_1, name='pool1')
                layer2_2 = lrelu(bn(conv2d(layer2_1, 128, [3, 3], initializer='random', name='conv_2'), is_training))

            with tf.variable_scope('g_ab_hidden3'):
                layer3_1 = max_pool(layer2_2, name='pool2')
                layer3_2 = lrelu(bn(conv2d(layer3_1, 256, [3, 3], initializer='random', name='conv_3'), is_training))

            with tf.variable_scope('g_ab_hidden4'):
                layer4_1 = conv2d_t(layer3_2, [None, 14, 14, 128], [2, 2], initializer='random', name='convT_4')
                layer4_2 = tf.concat([layer4_1, layer2_2], axis=3)
                layer4_3 = lrelu(bn(conv2d(layer4_2, 128, [3, 3], initializer='random', name='conv_4'), is_training))
                
            with tf.variable_scope('g_ab_hidden5'):
                layer5_1 = conv2d_t(layer4_3, [None, 28, 28, 64], [2, 2], initializer='random', name='convT5')
                layer5_2 = tf.concat([layer1_1, layer5_1], axis=3)
                layer5_3 = conv2d(layer5_2, 1, [3, 3], initializer='random', name='conv5')
                layer5_4 = conv2d(layer5_3, 1, [1, 1], initializer='random', name='conv6')
                gen_ab = tf.nn.sigmoid(layer5_4)

            return gen_ab, layer5_4


    def discriminator_B(self, inputs, is_training, reuse=False):
        with tf.variable_scope('discirminator_B', reuse=reuse):
            layer1 = lrelu(bn(conv2d(inputs, 64, [4, 4], strides=[1, 2, 2, 1], initializer='random', name='d_B_hiddne1'), is_training))
            layer2 = lrelu(bn(conv2d(layer1, 128, [4, 4], strides=[1, 2, 2, 1], initializer='random', name='d_B_hidden2'), is_training))
            flatten = tf.reshape(layer2, [-1, 7*7*128])
            layer3 = lrelu(bn(fully_connect(flatten, 1024, name='d_B_hidden3'), is_training))
            layer4 = fully_connect(layer3, 1, name='d_B_hidden')
            logits = sigmoid(layer4)

            return logits, layer4


    def generator_BA(self, inputs, is_training, reuse=False):
        with tf.variable_scope('Generator_BA', reuse=reuse):
            with tf.variable_scope('g_ba_hidden1'):
                layer1_1 = lrelu(bn(conv2d(inputs, 64, [3, 3], initializer='random', name='conv_1'), is_training))

            with tf.variable_scope('g_ba_hidden2'):
                layer2_1 = max_pool(layer1_1, name='pool1')
                layer2_2 = lrelu(bn(conv2d(layer2_1, 128, [3, 3], initializer='random', name='conv_2'), is_training))

            with tf.variable_scope('g_ba_hidden3'):
                layer3_1 = max_pool(layer2_2, name='pool2')
                layer3_2 = lrelu(bn(conv2d(layer3_1, 256, [3, 3], initializer='random', name='conv_3'), is_training))

            with tf.variable_scope('g_ba_hidden4'):
                layer4_1 = conv2d_t(layer3_2, [None, 14, 14, 128], [2, 2], initializer='random', name='convT_4')
                layer4_2 = tf.concat([layer4_1, layer2_2], axis=3)
                layer4_3 = lrelu(bn(conv2d(layer4_2, 128, [3, 3], initializer='random', name='conv_4'), is_training))
                
            with tf.variable_scope('g_ba_hidden5'):
                layer5_1 = conv2d_t(layer4_3, [None, 28, 28, 64], [2, 2], initializer='random', name='convT5')
                layer5_2 = tf.concat([layer1_1, layer5_1], axis=3)
                layer5_3 = conv2d(layer5_2, 1, [3, 3], initializer='random', name='conv5')
                layer5_4 = conv2d(layer5_3, 1, [1, 1], initializer='random', name='conv6')
                gen_ba = tf.nn.sigmoid(layer5_4)

            return gen_ba, layer5_4


    def discriminator_A(self, inputs, is_training, reuse=False):
        with tf.variable_scope('discirminator_A', reuse=reuse):
            layer1 = lrelu(bn(conv2d(inputs, 64, [4, 4], strides=[1, 2, 2, 1], initializer='random', name='d_A_hiddne1'), is_training))
            layer2 = lrelu(bn(conv2d(layer1, 128, [4, 4], strides=[1, 2, 2, 1], initializer='random', name='d_A_hidden2'), is_training))
            flatten = tf.reshape(layer2, [-1, 7*7*128])
            layer3 = lrelu(bn(fully_connect(flatten, 1024, name='d_A_hidden3'), is_training))
            layer4 = fully_connect(layer3, 1, name='d_A_hidden')
            logits = sigmoid(layer4)

            return logits, layer4


    def build_model(self):
        """
        input_a: 정상적인 MNIST
        input_b: 90도 회전시킨 MNIST
        is_train: training 여부(batch norm)
        """
        self.input_a = tf.placeholder(tf.float32, [None]+self.IMAGE_SHAPE, name='input_a')
        self.input_b = tf.placeholder(tf.float32, [None]+self.IMAGE_SHAPE, name='input_b')
        self.is_train = tf.placeholder(tf.bool)

        # ABA task
        g_AB, g_AB_net = self.generator_AB(self.input_a, self.is_train)
        d_real_B, d_real_B_net = self.discriminator_B(self.input_b, self.is_train)
        d_fake_B, d_fake_B_net = self.discriminator_B(g_AB, self.is_train, reuse=True)
        g_ABA, g_ABA_net = self.generator_BA(g_AB, self.is_train)

        # BAB task
        g_BA, g_BA_net = self.generator_BA(self.input_b, self.is_train, reuse=True)
        d_real_A, d_real_A_net = self.discriminator_A(self.input_a, self.is_train)
        d_fake_A, d_fake_A_net = self.discriminator_A(g_BA, self.is_train, reuse=True)
        g_BAB, g_BAB_net = self.generator_AB(g_BA, self.is_train, reuse=True)

        # reconstruction loss
        loss_reconst_A = tf.reduce_mean(tf.losses.mean_squared_error(self.input_a, g_ABA))
        loss_reconst_B = tf.reduce_mean(tf.losses.mean_squared_error(self.input_b, g_BAB))

        # discriminator loss
        loss_d_real_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_A_net), logits=d_real_A_net))
        loss_d_fake_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_A_net), logits=d_fake_A_net))

        loss_d_real_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_B_net), logits=d_real_B_net))
        loss_d_fake_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_B_net), logits=d_fake_B_net))

        # generator loss
        loss_g_ab = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_A), logits=d_fake_A))
        loss_g_ba = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_B), logits=d_fake_B))

        # final loss
        l_g_ab = loss_g_ab + loss_reconst_A
        l_d_b = loss_d_real_B + loss_d_fake_B 
        
        l_g_ba = loss_g_ba + loss_reconst_B
        l_d_a = loss_d_real_A + loss_d_fake_A

        self.G_loss = l_g_ab + l_g_ba
        self.D_loss = l_d_b + l_d_a

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE, beta1=self.BETA1).minimize(self.D_loss, var_list=d_vars)
            self.G_optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE*5, beta1=self.BETA1).minimize(self.G_loss, var_list=g_vars)

        self.d_loss_sum = tf.summary.merge([tf.summary.scalar("d_loss", self.D_loss)])
        self.g_loss_sum = tf.summary.merge([tf.summary.scalar("g_loss", self.G_loss)])

        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

        self.fakeAB, _ = self.generator_AB(self.input_a, is_training=False, reuse=True)
        # self.fakeABA, _ = self.generator_BA(self.fakeAB, is_training=False, reuse=True)

    def train_model(self):
        if not os.path.exists(self.MODEL_NAME+'_result'):   os.mkdir(self.MODEL_NAME+'_result')
        if not os.path.exists(self.LOGS_DIR):   os.mkdir(self.LOGS_DIR)
        if not os.path.exists(self.CKPT_DIR):   os.mkdir(self.CKPT_DIR)
        if not os.path.exists(self.OUTPUT_DIR): os.mkdir(self.OUTPUT_DIR)

        mnist = input_data.read_data_sets(self.DATASET_PATH, one_hot=True)
        ckpt_save_path = os.path.join(self.CKPT_DIR, self.MODEL_NAME+'_'+str(self.N_BATCH)+'_'+str(self.LEARNING_RATE))
        
        inputs = mnist.train.images
        half = int(mnist.train.num_examples / 2)

        domainA = inputs[:half].reshape(-1, 28, 28, 1)
        domainB = inputs[half:].reshape(-1, 28, 28, 1)
        domainB = scipy.ndimage.interpolation.rotate(domainB, 90, axes=(1, 2))

        sampleABA = domainA[:8]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            total_batch = int(len(domainA) / self.N_BATCH)
            counter = 0

            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.LOGS_DIR, sess.graph)

            for epoch in tqdm(range(self.N_EPOCH)):

                d_total_loss, g_total_loss = 0, 0

                for i in range(total_batch):
                    batch_domainA = domainA[i*self.N_BATCH : i*self.N_BATCH+self.N_BATCH]
                    batch_domainB = domainB[i*self.N_BATCH : i*self.N_BATCH+self.N_BATCH]

                    feed_dict = {self.input_a:batch_domainA, self.input_b:batch_domainB, self.is_train:True}
                    _, d_summary, d_loss = sess.run([self.D_optimizer, self.d_loss_sum, self.D_loss], feed_dict=feed_dict)
                    _, g_summary, g_loss = sess.run([self.G_optimizer, self.g_loss_sum, self.G_loss], feed_dict=feed_dict)

                    self.writer.add_summary(d_summary, global_step=counter)
                    self.writer.add_summary(g_summary, global_step=counter)
                    counter += 1

                    d_total_loss += d_loss
                    g_total_loss += g_loss

                samples = sess.run(self.fakeAB, feed_dict={self.input_a:sampleABA, self.is_train:False})
                samples = np.reshape(samples, [-1, 28, 28])
                save_path = os.path.join(self.OUTPUT_DIR, self.MODEL_NAME+'_'+str(epoch+1).zfill(3)+'.png')
                draw_plot_gan(samples, save_path)
                
                print('\nEPOCH: {}\t'.format(epoch+1), 'D_LOSS: {:.8}\t'.format(d_total_loss / total_batch), 'G_LOSS: {:.8}'.format(g_total_loss / total_batch))
                self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            
            self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            print('Finish save model')
                
