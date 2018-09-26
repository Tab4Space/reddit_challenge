import tensorflow as tf
import numpy as np
import os, random
import tensorflow.contrib.slim as slim

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, max_pool, lrelu, bn, relu
from utils import read_data_path, next_batch, read_image, read_annotation, draw_plot_segmentation

class PSPNET(object):
    MODEL = 'PSPNET'

    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'PSPNET'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.N_CLASS = 151
        self.RESIZE = 192
        
        self.TRAIN_IMAGE_PATH = './DATA/ADEChallengeData2016/images/training/'
        self.TRAIN_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/training/'

        self.VALID_IMAGE_PATH = './DATA/ADEChallengeData2016/images/validation/'
        self.VALID_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/validation/'


    def make_model(self, inputs, is_training):
        with tf.variable_scope('ResNet50'):
            x = conv2d(inputs, 64, [7, 7], strides=[1, 2, 2, 1], name='conv1')    # size 1/2
            x = bn(x, is_training)
            x = relu(x)
            x = max_pool(x, ksize=[1, 3, 3, 1], name='max_pool1')                        # size 1/4

            x = self.conv_block(x, [64, 64, 256], is_training, '2_1', s=1)
            x = self.identity_block(x, [64, 64, 256], is_training, '2_2')
            x = self.identity_block(x, [64, 64, 256], is_training, '2_3')

            x = self.conv_block(x, [128, 128, 512], is_training, '3_1')
            x = self.identity_block(x, [128, 128, 512], is_training, '3_2')
            x = self.identity_block(x, [128, 128, 512], is_training, '3_3')

            x = self.conv_block(x, [256, 256, 1024], is_training, '4_1')
            x = self.identity_block(x, [256, 256, 1024], is_training, '4_2')
            x = self.identity_block(x, [256, 256, 1024], is_training, '4_3')
            x = self.identity_block(x, [256, 256, 1024], is_training, '4_4')
            x = self.identity_block(x, [256, 256, 1024], is_training, '4_5')
            x = self.identity_block(x, [256, 256, 1024], is_training, '4_6')

            x = self.conv_block(x, [512, 512, 2048], is_training, '5_1')
            x = self.identity_block(x, [512, 512, 2048], is_training, '5_2')
            feature_map = self.identity_block(x, [512, 512, 2048], is_training, '5_3')        # size: (6, 6)

        with tf.variable_scope('Pyramid_Pool'):
            pool_1x1 = max_pool(feature_map, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], name='pool_1x1')
            pool_2x2 = max_pool(feature_map, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], name='pool_2x2')
            pool_3x3 = max_pool(feature_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool_3x3')
            pool_6x6 = max_pool(feature_map, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], name='pool_6x6')

            conv_1x1 = relu(bn(conv2d(pool_1x1, 512, [3, 3], name='conv_1x1'), is_training))   # reduce dimension
            conv_2x2 = relu(bn(conv2d(pool_2x2, 512, [3, 3], name='conv_2x2'), is_training))   # reduce dimension
            conv_3x3 = relu(bn(conv2d(pool_3x3, 512, [3, 3], name='conv_3x3'), is_training))   # reduce dimension
            conv_6x6 = relu(bn(conv2d(pool_6x6, 512, [3, 3], name='conv_6x6'), is_training))   # reduce dimension

            upconv_1x1 = tf.image.resize_bilinear(conv_1x1, [6, 6])
            upconv_2x2 = tf.image.resize_bilinear(conv_2x2, [6, 6])
            upconv_3x3 = tf.image.resize_bilinear(conv_3x3, [6, 6])
            upconv_6x6 = tf.image.resize_bilinear(conv_6x6, [6, 6])

            concated = tf.concat([feature_map, upconv_1x1, upconv_2x2, upconv_3x3, upconv_6x6], axis=3)

            out = relu(bn(conv2d(concated, 512, [3, 3], name='out1'), is_training))
            
            out = conv2d_t(out, [None, 12, 12, 256], [3, 3], name='out2')       # (12, 12)
            out = conv2d_t(out, [None, 24, 24, self.N_CLASS], [3, 3], name='out3')       # (24, 24)
            out = conv2d_t(out, [None, 48, 48, self.N_CLASS], [3, 3], name='out4')       # (24, 24)
            out = conv2d_t(out, [None, self.RESIZE, self.RESIZE, self.N_CLASS], [3, 3], name='out5', strides=[1, 4, 4, 1])       # (24, 24)

            pred = tf.argmax(out, axis=3)
            pred = tf.expand_dims(pred, dim=3)

            return out, pred

    def build_model(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.RESIZE, self.RESIZE, 3])         # images
        self.label_y = tf.placeholder(dtype=tf.int32, shape=[None, self.RESIZE, self.RESIZE, 1])         # annotations
        self.is_train = tf.placeholder(dtype=tf.bool)

        self.logits, self.pred = self.make_model(self.input_x, self.is_train)
        
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(self.label_y, [3])))
        
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])
    
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def train_model(self):
        if not os.path.exists(self.MODEL_NAME+'_result'):   os.mkdir(self.MODEL_NAME+'_result')
        if not os.path.exists(self.LOGS_DIR):   os.mkdir(self.LOGS_DIR)
        if not os.path.exists(self.CKPT_DIR):   os.mkdir(self.CKPT_DIR)
        if not os.path.exists(self.OUTPUT_DIR): os.mkdir(self.OUTPUT_DIR)
        
        train_set_path = read_data_path(self.TRAIN_IMAGE_PATH, self.TRAIN_LABEL_PATH)
        valid_set_path = read_data_path(self.VALID_IMAGE_PATH, self.VALID_LABEL_PATH)

        ckpt_save_path = os.path.join(self.CKPT_DIR, self.MODEL_NAME+'_'+str(self.N_BATCH)+'_'+str(self.LEARNING_RATE))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            total_batch = int(len(train_set_path) / self.N_BATCH)
            counter = 0

            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.LOGS_DIR, sess.graph)

            for epoch in range(self.N_EPOCH):
                total_loss = 0
                random.shuffle(train_set_path)           # 매 epoch마다 데이터셋 shuffling
                random.shuffle(valid_set_path)

                for i in range(int(len(train_set_path) / self.N_BATCH)):
                    # print(i)
                    batch_xs_path, batch_ys_path = next_batch(train_set_path, self.N_BATCH, i)
                    batch_xs = read_image(batch_xs_path, [self.RESIZE, self.RESIZE])
                    batch_ys = read_annotation(batch_ys_path, [self.RESIZE, self.RESIZE])

                    feed_dict = {self.input_x: batch_xs, self.label_y: batch_ys, self.is_train: True}

                    _, summary_str ,loss = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict=feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    counter += 1
                    total_loss += loss

                ## validation 과정
                valid_xs_path, valid_ys_path = next_batch(valid_set_path, 4, 0)
                valid_xs = read_image(valid_xs_path, [self.RESIZE, self.RESIZE])
                valid_ys = read_annotation(valid_ys_path, [self.RESIZE, self.RESIZE])
                
                valid_pred = sess.run(self.pred, feed_dict={self.input_x: valid_xs, self.label_y: valid_ys, self.is_train:False})
                valid_pred = np.squeeze(valid_pred, axis=3)
                
                valid_ys = np.squeeze(valid_ys, axis=3)

                ## plotting and save figure
                img_save_path = self.OUTPUT_DIR + '/' + str(epoch).zfill(3) + '.png'
                draw_plot_segmentation(img_save_path, valid_xs, valid_pred, valid_ys)

                print('\nEpoch:', '%03d' % (epoch + 1), 'Avg Loss: {:.6}\t'.format(total_loss / total_batch))
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

        