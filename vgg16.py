import tensorflow as tf
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from operation import conv2d, max_pool, fully_connect


class VGG16(object):
    def __init__(self, **kwargs):
        self.N_BATCH = kwargs['batch']
        self.N_EPOCH = kwargs['epoch']
        self.N_CLASS = kwargs['classes']
        self.LEARNING_RATE = kwargs['learning_rate']

        self.IMAGE_SHAPE = kwargs['image_shape']
        self.MODEL_NAME = kwargs['model_name']
        
        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')

        self.DATASET_PATH = './DATA/MNIST/'


    def make_model(self, inputs, keep_prob):
        conv1_1 = conv2d(inputs, 64, [3, 3], name='conv1_1')
        conv1_2 = conv2d(conv1_1, 64, [3, 3], name='conv1_2')
        pool1 = max_pool(conv1_2, name='pool1')

        conv2_1 = conv2d(pool1, 128, [3, 3], name='conv2_1')
        conv2_2 = conv2d(conv2_1, 128, [3, 3], name='conv2_2')
        pool2 = max_pool(conv2_2, name='pool2')
    
        conv3_1 = conv2d(pool2, 256, [3, 3], name='conv3_1')
        conv3_2 = conv2d(conv3_1, 256, [3, 3], name='conv3_2')
        conv3_3 = conv2d(conv3_2, 256, [3, 3], name='conv3_3')
        pool3 = max_pool(conv3_3, name='pool3')

        conv4_1 = conv2d(pool3, 512, [3, 3], name='conv4_1')
        conv4_2 = conv2d(conv4_1, 512, [3, 3], name='conv4_2')
        conv4_3 = conv2d(conv4_2, 512, [3, 3], name='conv4_3')
        pool4 = max_pool(conv4_3, name='pool4')

        conv5_1 = conv2d(pool4, 512, [3, 3], name='conv5_1')
        conv5_2 = conv2d(conv5_1, 512, [3, 3], name='conv5_2')
        conv5_3 = conv2d(conv5_2, 512, [3, 3], name='conv5_3')
        pool5 = max_pool(conv5_3, name='pool5')

        flatten = tf.reshape(pool5, shape=[self.N_BATCH, -1], name='flatten')
        fc1 = fully_connect(flatten, 4096, name='fc1')
        fc1_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob, name='fc1_dropout')

        fc2 = fully_connect(fc1_dropout, 4096, name='fc2')
        fc2_dropout = tf.nn.dropout(fc2, keep_prob=keep_prob, name='fc2_dropout')

        logits = fully_connect(fc2_dropout, self.N_CLASS, name='fc3')
        
        return logits


    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[self.N_BATCH]+self.IMAGE_SHAPE)
        self.LABEL_Y = tf.placeholder(dtype=tf.float32, shape=[self.N_BATCH, self.N_CLASS])
        self.KEEP_PROB = tf.placeholder(dtype=tf.float32)
        
        self.pred = self.make_model(self.INPUT_X, self.KEEP_PROB)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.LABEL_Y))

        self.optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])

    
    def train_model(self):
        if not os.path.exists(self.MODEL_NAME+'_result'):   os.mkdir(self.MODEL_NAME+'_result')
        if not os.path.exists(self.LOGS_DIR):   os.path.exists(self.LOGS_DIR)
        if not os.path.exists(self.CKPT_DIR):   os.path.exists(self.CKPT_DIR)
        if not os.path.exists(self.OUTPUT_DIR): os.path.exists(self.OUTPUT_DIR)

        mnist = input_data.read_data_sets(self.DATASET_PATH, one_hot=True)

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

                    feed_dict = {self.INPUT_X: batch_xs, self.LABEL_Y: batch_ys, self.KEEP_PROB: 0.7}
                    _, summary, loss = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict=feed_dict)
                    self.writer.add_summary(summary, counter)
                    counter += 1

                    total_loss += loss

                print('Epoch:', '%03d' % (epoch + 1), 'AVG Loss: ', '{:.6f}'.format(total_loss / total_batch))

                self.saver.save(self.CKPT_DIR, global_step=counter)
            
            self.saver.save(self.CKPT_DIR, global_step=counter)