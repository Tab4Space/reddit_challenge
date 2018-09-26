import tensorflow as tf
import numpy as np
import random, os
import tensorflow.contrib.slim as slim

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, atrous_conv2d, relu, bn, max_pool
from utils import read_data_path, next_batch, read_image, read_annotation, draw_plot_segmentation


class DeepLab_v3(object):
    MODEL = 'DeepLab_v3'

    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'DeepLab_v3'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.N_CLASS = 151
        self.RESIZE = 192
        
        self.TRAIN_IMAGE_PATH = './DATA/ADEChallengeData2016/images/training/'
        self.TRAIN_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/training/'

        self.VALID_IMAGE_PATH = './DATA/ADEChallengeData2016/images/validation2/'
        self.VALID_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/validation2/'
    
    def make_model(self, inputs, is_training):
        """
            extract feature using ResNet. Encoder

        """
        with tf.variable_scope('ResNet50'):
            x = conv2d(inputs, 64, [7, 7], strides=[1, 2, 2, 1], name='conv1')    # size 1/2
            x = bn(x, is_training)
            x = relu(x)
            x = max_pool(x, ksize=[1, 3, 3, 1], name='pool1')                        # size 1/4

            x = self.conv_block(x, [64, 64, 256], '2_1', is_training, s=1)
            x = self.identity_block(x, [64, 64, 256], '2_2', is_training)
            x = self.identity_block(x, [64, 64, 256], '2_3', is_training)

            x = self.conv_block(x, [128, 128, 512], '3_1', is_training)
            x = self.identity_block(x, [128, 128, 512], '3_2', is_training)
            x = self.identity_block(x, [128, 128, 512], '3_3', is_training)

            x = self.atrous_conv_block(x, [256, 256, 1024], '4_1', 2, is_training, s=1)
            x = self.atrous_identity_block(x, [256, 256, 1024], '4_2', 2, is_training)
            x = self.atrous_identity_block(x, [256, 256, 1024], '4_3', 2, is_training)
            x = self.atrous_identity_block(x, [256, 256, 1024], '4_4', 2, is_training)
            x = self.atrous_identity_block(x, [256, 256, 1024], '4_5', 2, is_training)
            x = self.atrous_identity_block(x, [256, 256, 1024], '4_6', 2, is_training)

            x = self.atrous_conv_block(x, [512, 512, 2048], '5_1', 4, is_training, s=1)
            x = self.atrous_identity_block(x, [512, 512, 2048], '5_2', 4, is_training)
            x = self.atrous_identity_block(x, [512, 512, 2048], '5_3', 4, is_training)



        """
            Astrous Pyrimid Pooling. Decoder
        """
        with tf.variable_scope('ASPP'):
            feature_map_shape = x.get_shape().as_list()

            # global average pooling
            # feature 맵의 height, width를 평균을 낸다.
            feature_map = tf.reduce_mean(x, [1, 2], keepdims=True)

            feature_map = conv2d(feature_map, 256, [1, 1], name='gap_feature_map')
            feature_map = tf.image.resize_bilinear(feature_map, [feature_map_shape[1], feature_map_shape[2]])

            rate1 = conv2d(x, 256, [1, 1], name='rate1')
            rate6 = atrous_conv2d(x, 256, [3, 3], rate=6, name='rate6')
            rate12 = atrous_conv2d(x, 256, [3, 3], rate=12, name='rate12')
            rate18 = atrous_conv2d(x, 256, [3, 3], rate=18, name='rate18')

            concated = tf.concat([feature_map, rate1, rate6, rate12, rate18], axis=3)

            net = conv2d(concated, 256, [1, 1], name='net')

            logits = conv2d(net, self.N_CLASS, [1, 1], name='logits')
            logits = tf.image.resize_bilinear(logits, size=[self.RESIZE, self.RESIZE], name='out')

            pred = tf.argmax(logits, axis=3)
            pred = tf.expand_dims(pred, dim=3)

            return logits, pred


    def build_model(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.RESIZE, self.RESIZE, 3])
        self.label_y = tf.placeholder(dtype=tf.int32, shape=[None, self.RESIZE, self.RESIZE, 1])
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


    def identity_block(self, inputs, filters, stage, is_training):
        filter1, filter2, filter3 = filters
        layer1 = relu(bn(conv2d(inputs, filter1, [1, 1], name=stage+'_a_identity', padding='VALID'), is_training))
        layer2 = relu(bn(conv2d(layer1, filter2, [3, 3], name=stage+'_b_identity'), is_training))
        layer3 = bn(conv2d(layer2, filter3, [1, 1], name=stage+'_c_identity', padding='VALID'), is_training)
        layer4 = relu(tf.add(layer3, inputs))
        return layer4


    def conv_block(self, inputs, depths, stage, is_training, s=2):
        depth1, depth2, depth3 = depths
        layer1 = relu(bn(conv2d(inputs, depth1, [1, 1], name=stage+'_a_conv', strides=[1, s, s, 1], padding='VALID'), is_training))
        layer2 = relu(bn(conv2d(layer1, depth2, [3, 3], name=stage+'_b_conv'), is_training))
        layer3 = bn(conv2d(layer2, depth3, [1, 1], name=stage+'_c_conv', padding='VALID'), is_training)
        shortcut = bn(conv2d(inputs, depth3, [1, 1], name=stage+'_shortcut', strides=[1, s, s, 1], padding='VALID'), is_training)
        layer4 = relu(tf.add(layer3, shortcut))
        return layer4

        
    def atrous_identity_block(self, inputs, depths, stage, rate, is_training):
        depth1, depth2, depth3 = depths
        layer1 = relu(bn(atrous_conv2d(inputs, depth1, [1, 1], rate, name=stage+'_a_identity'), is_training))
        layer2 = relu(bn(atrous_conv2d(layer1, depth2, [3, 3], rate, name=stage+'_b_identity'), is_training))
        layer3 = bn(atrous_conv2d(layer2, depth3, [1, 1], rate, name=stage+'_c_identity'), is_training)
        layer4 = relu(tf.add(layer3, inputs))
        return layer4


    def atrous_conv_block(self, inputs, depths, stage, rate, is_training, s=2):
        depth1, depth2, depth3 = depths
        layer1 = relu(bn(atrous_conv2d(inputs, depth1, [1, 1], rate, name=stage+'_a_conv'), is_training))
        layer2 = relu(bn(atrous_conv2d(layer1, depth2, [3, 3], rate, name=stage+'_b_conv'), is_training))
        layer3 = bn(atrous_conv2d(layer2, depth3, [1, 1], rate, name=stage+'_c_conv'), is_training)
        shortcut = bn(conv2d(inputs, depth3, [1, 1], name=stage+'_shortcut', strides=[1, s, s, 1], padding='VALID'), is_training)
        layer4 = relu(tf.add(layer3, shortcut))
        return layer4
        

