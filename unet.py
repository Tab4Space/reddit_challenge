import tensorflow as tf
import numpy as np
import os, random
import tensorflow.contrib.slim as slim

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, max_pool, lrelu, bn, relu
from utils import read_data_path, next_batch, read_image, read_annotation, draw_plot_segmentation

class UNET(object):
    MODEL = 'UNET'

    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'UNET'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.N_CLASS = 151
        self.RESIZE = 224
        
        self.TRAIN_IMAGE_PATH = './DATA/ADEChallengeData2016/images/training2/'
        self.TRAIN_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/training2/'

        self.VALID_IMAGE_PATH = './DATA/ADEChallengeData2016/images/validation2/'
        self.VALID_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/validation2/'
        

    def make_model(self, inputs, is_training):
        with tf.variable_scope('ENCODER'):
            layer1_1 = conv2d(inputs, 32, [3, 3], name='layer1_1')
            layer1_2 = conv2d(layer1_1, 32, [3, 3], name='layer1_2')
            layer1_3 = max_pool(layer1_2, name='layer1_3')           # original image 1/2, (112, 112)
            
            layer2_1 = relu(bn(conv2d(layer1_3, 64, [3, 3], name='layer2_1'), is_training))
            layer2_2 = relu(bn(conv2d(layer2_1, 64, [3, 3], name='layer2_2'), is_training))
            layer2_3 = max_pool(layer2_2, name='layer2_3')           # original image 1/4, (56, 56)

            layer3_1 = relu(bn(conv2d(layer2_3, 128, [3, 3], name='layer3_1'), is_training))
            layer3_2 = relu(bn(conv2d(layer3_1, 128, [3, 3], name='layer3_2'), is_training))
            layer3_3 = max_pool(layer3_2, name='layer3_3')           # original image 1/8, (28, 28)
            
            layer4_1 = relu(bn(conv2d(layer3_3, 256, [3, 3], name='layer4_1'), is_training))
            layer4_2 = relu(bn(conv2d(layer4_1, 256, [3, 3], name='layer4_2'), is_training))
            layer4_3 = max_pool(layer4_2, name='layer4_3')           # original image 1/16, (14, 14)

            layer5_1 = relu(bn(conv2d(layer4_3, 512, [3, 3], name='layer5_1'), is_training))
            layer5_2 = relu(bn(conv2d(layer5_1, 512, [3, 3], name='layer5_2'), is_training))

        with tf.variable_scope('DECODER'):
            layer6_1 = relu(bn(conv2d_t(layer5_2, [None, 28, 28, 256], [2, 2], name='layer6_1'), is_training))
            layer6_2 = tf.concat([layer4_2, layer6_1], axis=3, name='layer6_2')
            layer6_3 = relu(bn(conv2d(layer6_2, 256, [3, 3], name='layer6_3'), is_training))
            layer6_4 = relu(bn(conv2d(layer6_3, 256, [3, 3], name='layer6_4'), is_training))

            l6_4_shape = layer6_4.get_shape()
            layer7_1 = relu(bn(conv2d_t(layer6_4, [None, 56, 56, 128], [2, 2], name='layer7_1'), is_training))
            layer7_2 = tf.concat([layer3_2, layer7_1], axis=3, name='layer7_2')
            layer7_3 = relu(bn(conv2d(layer7_2, 128, [3, 3], name='layer7_2'), is_training))
            layer7_4 = relu(bn(conv2d(layer7_3, 128, [3, 3], name='layer7_3'), is_training))

            l7_4_shape = layer7_4.get_shape()
            layer8_1 = relu(bn(conv2d_t(layer7_4, [None, 112, 112, 64], [2, 2], name='layer8_1'), is_training))
            layer8_2 = tf.concat([layer2_2, layer8_1], axis=3, name='layer8_2')
            layer8_3 = relu(bn(conv2d(layer8_2, 64, [3, 3], name='layer8_3'), is_training))
            layer8_4 = relu(bn(conv2d(layer8_3, 64, [3, 3], name='layer8_4'), is_training))

            l8_4_shape = layer8_4.get_shape()
            layer9_1 = relu(bn(conv2d_t(layer8_4, [None, 224, 224, 32], [2, 2], name='layer9_1'), is_training))
            layer9_2 = tf.concat([layer1_2, layer9_1], axis=3, name='layer9_2')
            layer9_3 = relu(bn(conv2d(layer9_2, self.N_CLASS, [3, 3], name='layer9_3'), is_training))
            layer9_4 = relu(bn(conv2d(layer9_3, self.N_CLASS, [3, 3], name='layer9_4'), is_training))

            logits = conv2d(layer9_4, self.N_CLASS, [1, 1], name='logits')
            annot_pred = tf.argmax(logits, axis=3)
            expand_pred = tf.expand_dims(annot_pred, dim=3)

            return logits, expand_pred, layer5_2


    def build_model(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.RESIZE, self.RESIZE, 3])
        self.label_y = tf.placeholder(dtype=tf.int32, shape=[None, self.RESIZE, self.RESIZE, 1])
        self.is_train = tf.placeholder(dtype=tf.bool)

        self.logits, self.pred, _ = self.make_model(self.input_x, self.is_train)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(self.label_y, [3])))

        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('UNET_loss', self.loss)])
        
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

            for epoch in tqdm(range(self.N_EPOCH)):
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
                valid_xs_path, valid_ys_path = next_batch(valid_set_path, self.N_BATCH, epoch)
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
            
