import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim

from tqdm import tqdm
from custom_op import conv2d, relu, fully_connect, bn, max_pool, avg_pool
from tensorflow.examples.tutorials.mnist import input_data

class ResNet50(object):
    MODEL = 'ResNet50'

    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'ResNet50'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.N_CLASS = 10
        self.IMAGE_SHAPE = [28, 28, 1]
        self.RESIZE = 224
        
        self.DATASET_PATH = './DATA/MNIST/'


    def make_model(self, inputs, is_training):

        with tf.variable_scope('STAGE_1'):
            layer = relu(bn(conv2d(inputs, 64, [7, 7], strides=[1, 2, 2, 1], name='initial_block'), is_training))
            layer = max_pool(layer)

        with tf.variable_scope('STAGE_2'):
            layer = self.conv_block(layer, [64, 64, 256], is_training, 'a', s=1)
            layer = self.identity_block(layer, [64, 64, 256], is_training, 'b')
            layer = self.identity_block(layer, [64, 64, 256], is_training, 'c')

        with tf.variable_scope('STAGE_3'):
            layer = self.conv_block(layer, [128, 128, 512], is_training, 'a')
            layer = self.identity_block(layer, [128, 128, 512], is_training, 'b')
            layer = self.identity_block(layer, [128, 128, 512], is_training, 'c')

        with tf.variable_scope('STAGE_4'):
            layer = self.conv_block(layer, [256, 256, 1024], is_training, 'a')
            layer = self.identity_block(layer, [256, 256, 1024], is_training, 'b')
            layer = self.identity_block(layer, [256, 256, 1024], is_training, 'c')
            layer = self.identity_block(layer, [256, 256, 1024], is_training, 'd')
            layer = self.identity_block(layer, [256, 256, 1024], is_training, 'e')
            layer = self.identity_block(layer, [256, 256, 1024], is_training, 'f')

        with tf.variable_scope('STAGE_5'):
            layer = self.conv_block(layer, [512, 512, 2048], is_training, 'a')
            layer = self.identity_block(layer, [512, 512, 2048], is_training, 'b')
            layer = self.identity_block(layer, [512, 512, 2048], is_training, 'c')

        with tf.variable_scope('FINAL_STAGE'):
            layer = avg_pool(layer, [1, 7, 7, 1], [1, 1, 1, 1], padding='VALID')

            _, h, w, d = layer.get_shape().as_list()

            layer = tf.reshape(layer, [-1, h*w*d])
            layer = fully_connect(layer, self.N_CLASS, 'fc')

            return layer


    def build_model(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None]+self.IMAGE_SHAPE)
        self.resize_x = tf.image.resize_images(self.input_x, size=[self.RESIZE, self.RESIZE])
        self.label_y = tf.placeholder(dtype=tf.float32, shape=[None, self.N_CLASS])
        self.is_train = tf.placeholder(dtype=tf.bool)

        self.logits = self.make_model(self.resize_x, self.is_train)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label_y))

        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])

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
                total_loss = 0
            
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(self.N_BATCH)
                    batch_xs = np.reshape(batch_xs, [self.N_BATCH]+self.IMAGE_SHAPE)

                    feed_dict = {self.input_x: batch_xs, self.label_y: batch_ys, self.is_train: True}
                    _, summary, loss = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict=feed_dict)

                    self.writer.add_summary(summary, counter)
                    counter += 1

                    total_loss += loss

                print('Epoch:', '%03d' % (epoch + 1), 'AVG Loss: ', '{:.6f}'.format(total_loss / total_batch))

                self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            
            self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            print('Finish save model')


    def identity_block(self, inputs, depths, is_training, stage):
        depth1, depth2, depth3 = depths
        layer1 = relu(bn(conv2d(inputs, depth1, [1, 1], padding='VALID', name=stage+'_layer1'), is_training))
        layer2 = relu(bn(conv2d(layer1, depth2, [3, 3], name=stage+'_layer2'), is_training))
        layer3 = relu(bn(conv2d(layer2, depth3, [1, 1], padding='VALID', name=stage+'_layer3'), is_training))
        layer4 = relu(tf.add(layer3, inputs, name=stage+'_layer4'))
        return layer4
        

    def conv_block(self, inputs, depths, is_training, stage, s=2):
        depth1, depth2, depth3 = depths
        layer1 = relu(bn(conv2d(inputs, depth1, [1, 1], strides=[1, s, s, 1], padding='VALID', name=stage+'_layer1'), is_training))
        layer2 = relu(bn(conv2d(layer1, depth2, [3, 3], name=stage+'_layer2'), is_training))
        layer3 = bn(conv2d(layer2, depth3, [1, 1], padding='VALID', name=stage+'_layer3'), is_training)
        shortcut = bn(conv2d(inputs, depth3, [1, 1], strides=[1, s, s, 1], padding='VALID', name=stage+'_shortcut'), is_training)
        layer4 = relu(tf.add(layer3, shortcut, name=stage+'_layer4'))
        return layer4


