import tensorflow as tf
import numpy as np
from custom_op import conv2d, max_pool, fully_connect
from tqdm import tqdm


class VGG16(object):
    def __init__(self):
        self.N_BATCH
        self.N_EPOCH
        self.L_RATE
        self.N_CLASS


    def model(self, inputs):
        conv1_1 = conv2d(inputs, 64, 3, 3, name='conv1_1')
        conv1_2 = conv2d(conv1_1, 64, 3, 3, name='conv1_2')
        pool1 = max_pool(conv1_2, name='pool1')

        conv2_1 = conv2d(pool1, 128, 3, 3, name='conv2_1')
        conv2_2 = conv2d(conv2_1, 128, 3, 3, name='conv2_2')
        pool2 = max_pool(conv2_2, name='pool2')
    
        covn3_1 = conv2d(pool2, 256, 3, 3, name='conv3_1')
        conv3_2 = conv2d(conv3_1, 256, 3, 3, name='conv3_2')
        conv3_3 = conv2d(conv3_2, 256, 3, 3, name='conv3_3')
        pool3 = max_pool(conv3_3, name='pool3')


        covn4_1 = conv2d(pool3, 512, 3, 3, name='conv4_1')
        conv4_2 = conv2d(conv4_1, 512, 3, 3, name='conv4_2')
        conv4_3 = conv2d(conv4_2, 512, 3, 3, name='conv4_3')
        pool4 = max_pool(conv4_3, name='pool4')


        covn5_1 = conv2d(pool4, 512, 3, 3, name='conv5_1')
        conv5_2 = conv2d(conv5_1, 512, 3, 3, name='conv5_2')
        conv5_3 = conv2d(conv5_2, 512, 3, 3, name='conv5_3')
        pool5 = max_pool(conv5_3, name='pool5')

        flatten = tf.reshape(pool5, shape=[self.batch, -1], name='flatten')
        fc1 = fully_connect(flatten, 4096, name='FC1')
        fc1_dropout = tf.nn.dropout(fc1, keep_prob=, name=)

        fc2 = fully_connect(fc1_dropout, 4096, name='FC2')
        fc2_dropout = tf.nn.dropout(fc2, keep_prob=, name=)

        fc3 = fully_connect(fc2_dropout, 1000, name='FC3')
        logits = tf.nn.softmax(fc3, name='softmax')
        
        return logits

    def build_model(self):
        self.INPUT_X = tf.placeholder(dtye=tf.float32, shape=[])
        self.INPUT_Y = tf.placeholder(dtype=tf.floa32, shape=[])

        self.prediction = self.model(self.INPUT_X)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.INPUT_X, logits=self.prediction)

        self.optimizer = tf.train.GradientDescentOptimizer().minimize()


    def train_model(self):
        
        pass

    def test_model(self):
        pass