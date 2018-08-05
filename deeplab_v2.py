import tensorflow as tf
import numpy as np
import random
import os

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, relu, bn, max_pool, atrous_conv2d
from utils import read_image, load_data_path, next_batch


def identity_block(inputs, filters, stage, is_training=True):
    filter1, filter2, filter3 = filters
    layer1 = relu(bn(conv2d(inputs, filter1, 1, 1, name=stage+'_a_identity', padding='VALID'), is_training))
    layer2 = relu(bn(conv2d(layer1, filter2, 3, 3, name=stage+'_b_identity'), is_training))
    layer3 = bn(conv2d(layer2, filter3, 1, 1, name=stage+'_c_identity', padding='VALID'), is_training)
    layer4 = relu(tf.add(layer3, inputs))

    return layer4

def conv_block(inputs, filters, stage, s=2, is_training=True):
    filter1, filter2, filter3 = filters
    layer1 = relu(bn(conv2d(inputs, filter1, 1, 1, name=stage+'_a_conv', strides=[1, s, s, 1], padding='VALID'), is_training))
    layer2 = relu(bn(conv2d(layer1, filter2, 3, 3, name=stage+'_b_conv'), is_training))
    layer3 = bn(conv2d(layer2, filter3, 1, 1, name=stage+'_c_conv', padding='VALID'), is_training)
    shortcut = bn(conv2d(inputs, filter3, 1, 1, name=stage+'_shortcut', strides=[1, s, s, 1], padding='VALID'), is_training)
    layer4 = relu(tf.add(layer3, shortcut))
    
    return layer4

def atrous_identity_block(inputs, filters, stage, rate, is_training=True):
    filter1, filter2, filter3 = filters
    layer1 = relu(bn(atrous_conv2d(inputs, filter1, 1, 1, rate, name=stage+'_a_identity'), is_training))
    layer2 = relu(bn(atrous_conv2d(layer1, filter2, 3, 3, rate, name=stage+'_b_identity'), is_training))
    layer3 = bn(atrous_conv2d(layer2, filter3, 1, 1, rate, name=stage+'_c_identity'), is_training)
    layer4 = relu(tf.add(layer3, inputs))

    return layer4

def atrous_conv_block(inputs, filters, stage, rate, s=2, is_training=True):
    filter1, filter2, filter3 = filters
    layer1 = relu(bn(atrous_conv2d(inputs, filter1, 1, 1, rate, name=stage+'_a_conv'), is_training))
    layer2 = relu(bn(atrous_conv2d(layer1, filter2, 3, 3, rate, name=stage+'_b_conv'), is_training))
    layer3 = bn(atrous_conv2d(layer2, filter3, 1, 1, rate, name=stage+'_c_conv'), is_training)
    shortcut = bn(conv2d(inputs, filter3, 1, 1, name=stage+'_shortcut', strides=[1, s, s, 1], padding='VALID'), is_training)
    layer4 = relu(tf.add(layer3, shortcut))
    
    return layer4


class DeepLab_v2(object):
    def __init__(self):
        self.N_BATCH = 2
        self.N_EPOCH = 1000
        self.N_CLASS = 151
        self.L_RATE = 1e-5

        self.IMAGE_DATA_DIR = '../dataset/images/'
        self.ANNOTATION_DATA_DIR = '../dataset/annotations/'
        self.LOG_DIR = './logs/DeepLab_v2/'
        self.MODEL_NAME = 'DeepLab_v2'

    
    def model(self, inputs, is_training=True):
        """
            extract feature using ResNet. Encoder

        """
        with tf.variable_scope('ResNet50'):
            x = conv2d(inputs, 64, 7, 7, name='conv1', strides=[1, 2, 2, 1])    # size 1/2
            x = bn(x, is_training=True)
            x = relu(x)
            x = max_pool(x, 'pool1', ksize=[1, 3, 3, 1])                        # size 1/4
            print('1: ', x)

            x = conv_block(x, [64, 64, 256], '2_1', s=1)
            x = identity_block(x, [64, 64, 256], '2_2')
            x = identity_block(x, [64, 64, 256], '2_3')
            print('2: ', x)

            x = conv_block(x, [128, 128, 512], '3_1')
            x = identity_block(x, [128, 128, 512], '3_2')
            x = identity_block(x, [128, 128, 512], '3_3')
            print('3: ', x)

            x = atrous_conv_block(x, [256, 256, 1024], '4_1', rate=2, s=1)
            x = atrous_identity_block(x, [256, 256, 1024], '4_2', rate=2)
            x = atrous_identity_block(x, [256, 256, 1024], '4_3', rate=2)
            x = atrous_identity_block(x, [256, 256, 1024], '4_4', rate=2)
            x = atrous_identity_block(x, [256, 256, 1024], '4_5', rate=2)
            x = atrous_identity_block(x, [256, 256, 1024], '4_6', rate=2)
            print('4: ', x)

            x = atrous_conv_block(x, [512, 512, 2048], '5_1', rate=4, s=1)
            x = atrous_identity_block(x, [512, 512, 2048], '5_2', rate=4)
            x = atrous_identity_block(x, [512, 512, 2048], '5_3', rate=4)
            print('5: ', x)


        """
            Astrous Pyrimid Pooling. Decoder
        """
        with tf.variable_scope('ASPP'):
            rate6 = atrous_conv2d(x, self.N_CLASS, 3, 3, 6, name='rate6')
            rate6 = conv2d(rate6, self.N_CLASS, 1, 1, name='rate6_conv1')
            rate6 = conv2d(rate6, self.N_CLASS, 1, 1, name='rate6_conv2')

            rate12 = atrous_conv2d(x, self.N_CLASS, 3, 3, 12, name='rate12')
            rate12 = conv2d(rate12, self.N_CLASS, 1, 1, name='rate12_conv1')
            rate12 = conv2d(rate12, self.N_CLASS, 1, 1, name='rate12_conv2')

            rate18 = atrous_conv2d(x, self.N_CLASS, 3, 3, 18, name='rate18')
            rate18 = conv2d(rate18, self.N_CLASS, 1, 1, name='rate18_conv1')
            rate18 = conv2d(rate18, self.N_CLASS, 1, 1, name='rate18_conv2')
            
            rate24 = atrous_conv2d(x, self.N_CLASS, 3, 3, 24, name='rate24')
            rate24 = conv2d(rate24, self.N_CLASS, 1, 1, name='rate24_conv1')
            rate24 = conv2d(rate24, self.N_CLASS, 1, 1, name='rate24_conv2')

            self.logits = tf.add_n([rate6, rate12, rate18, rate24])
            self.out = tf.image.resize_bilinear(self.logits, size=[192, 192])
            print(self.logits)
            print(self.out)

            return self.logits, self.out

    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[None, 192, 192, 3])
        self.INPUT_Y = tf.placeholder(dtype=tf.int32, shape=[None, 192, 192, 1])

        self.logits, self.out = self.model(self.INPUT_X)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=tf.squeeze(self.INPUT_Y, [3])))
        self.optimizer = tf.train.AdamOptimizer(self.L_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])
        

    
    def train_model(self):
        data_set_path = load_data_path(self.IMAGE_DATA_DIR, self.ANNOTATION_DATA_DIR, 'training')
        valid_set_path = load_data_path(self.IMAGE_DATA_DIR, self.ANNOTATION_DATA_DIR, 'validation')
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.N_EPOCH):
                random.shuffle(data_set_path)           # 매 epoch마다 데이터셋 shuffling

                for i in range(int(len(data_set_path) / 2)):
                    batch_img_path, batch_ann_path = next_batch(data_set_path, i, self.N_BATCH)
                    batch_imgs, batch_anns = read_image(batch_img_path, batch_ann_path, self.N_BATCH, 192, 192)

                    _ ,loss_val = sess.run([self.optimizer, self.loss], feed_dict={self.INPUT_X:batch_imgs, self.INPUT_Y:batch_anns})
                    # self.writer.add_summary(summary_str, counter)
                    # counter += 1

                print('EPOCH: {}\t'.format(epoch+1), 'LOSS: {:.8}\t'.format(loss_val))
        



model = DeepLab_v2()
model.build_model()
model.train_model()