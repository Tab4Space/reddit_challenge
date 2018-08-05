import tensorflow as tf
import numpy as np
from custom_op import conv2d, relu, fully_connect, bn, max_pool, avg_pool, fully_connect, conv2d_t
from utils import next_batch, load_data_path, read_image
import random


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


class PSPNET(object):
    def __init__(self):
        self.N_BATCH = 2
        self.N_EPOCH = 100
        self.N_CLASS = 151
        self.L_RATE = 1e-5

        self.IMAGE_DATA_DIR = '../dataset/images/'
        self.ANNOTATION_DATA_DIR = '../dataset/annotations/'
        self.LOG_DIR = './logs/PSPNET/'
        self.MODEL_NAME = 'PSPNET'


    def model(self, inputs):
        with tf.variable_scope('ResNet50'):
            x = conv2d(inputs, 64, 7, 7, name='conv1', strides=[1, 2, 2, 1])    # size 1/2
            x = bn(x, is_training=True)
            x = relu(x)
            x = max_pool(x, 'pool1', ksize=[1, 3, 3, 1])                        # size 1/4

            x = conv_block(x, [64, 64, 256], '2_1', s=1)
            x = identity_block(x, [64, 64, 256], '2_2')
            x = identity_block(x, [64, 64, 256], '2_3')

            x = conv_block(x, [128, 128, 512], '3_1')
            x = identity_block(x, [128, 128, 512], '3_2')
            x = identity_block(x, [128, 128, 512], '3_3')

            x = conv_block(x, [256, 256, 1024], '4_1')
            x = identity_block(x, [256, 256, 1024], '4_2')
            x = identity_block(x, [256, 256, 1024], '4_3')
            x = identity_block(x, [256, 256, 1024], '4_4')
            x = identity_block(x, [256, 256, 1024], '4_5')
            x = identity_block(x, [256, 256, 1024], '4_6')

            x = conv_block(x, [512, 512, 2048], '5_1')
            x = identity_block(x, [512, 512, 2048], '5_2')
            feature_map = identity_block(x, [512, 512, 2048], '5_3')        # size: (6, 6)

        with tf.variable_scope('Pyramid_Pool'):
            pool_1x1 = max_pool(feature_map, 'pool_1x1', ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1])
            pool_2x2 = max_pool(feature_map, 'pool_2x2', ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1])
            pool_3x3 = max_pool(feature_map, 'pool_3x3', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
            pool_6x6 = max_pool(feature_map, 'pool_6x6', ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1])

            conv_1x1 = relu(bn(conv2d(pool_1x1, 512, 3, 3, 'conv_1x1'), is_training=True))   # reduce dimension
            conv_2x2 = relu(bn(conv2d(pool_2x2, 512, 3, 3, 'conv_2x2'), is_training=True))   # reduce dimension
            conv_3x3 = relu(bn(conv2d(pool_3x3, 512, 3, 3, 'conv_3x3'), is_training=True))   # reduce dimension
            conv_6x6 = relu(bn(conv2d(pool_6x6, 512, 3, 3, 'conv_6x6'), is_training=True))   # reduce dimension

            upconv_1x1 = tf.image.resize_bilinear(conv_1x1, [6, 6])
            upconv_2x2 = tf.image.resize_bilinear(conv_2x2, [6, 6])
            upconv_3x3 = tf.image.resize_bilinear(conv_3x3, [6, 6])
            upconv_6x6 = tf.image.resize_bilinear(conv_6x6, [6, 6])

            concated = tf.concat([feature_map, upconv_1x1, upconv_2x2, upconv_3x3, upconv_6x6], axis=3)

            out = relu(bn(conv2d(concated, 512, 3, 3, 'out1'), is_training=True))
            
            out = conv2d_t(out, [self.N_BATCH, 12, 12, 256], 3, 3, 'out2')       # (12, 12)
            out = conv2d_t(out, [self.N_BATCH, 24, 24, 151], 3, 3, 'out3')       # (24, 24)
            out = conv2d_t(out, [self.N_BATCH, 48, 48, 151], 3, 3, 'out4')       # (24, 24)
            out = conv2d_t(out, [self.N_BATCH, 192, 192, 151], 3, 3, 'out5', strides=[1, 4, 4, 1])       # (24, 24)

            return out

    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[None, 192, 192, 3])
        self.INPUT_Y = tf.placeholder(dtype=tf.int32, shape=[None, 192, 192, 1])

        self.logits = self.model(self.INPUT_X)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(self.INPUT_Y, [3])))
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


model = PSPNET()
model.build_model()
model.train_model()