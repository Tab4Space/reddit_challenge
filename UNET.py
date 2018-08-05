import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from custom_op import conv2d, conv2d_t, max_pool, relu, bn
from utils import load_data_path, next_batch, read_image
import scipy.misc
import random
import os


class UNET(object):
    def __init__(self):
        self.N_BATCH = 2
        self.N_EPOCH = 1000
        self.N_CLASS = 151
        self.L_RATE = 1e-5

        self.IMAGE_DATA_DIR = '../dataset/images/'
        self.ANNOTATION_DATA_DIR = '../dataset/annotations/'
        self.LOG_DIR = './logs/UNET/'
        self.MODEL_NAME = 'UNET'
        
    def model(self, inputs, is_training=True):
        with tf.variable_scope('ENCODER'):
            layer1_1 = conv2d(inputs, 32, 3, 3, 'layer1_1')
            layer1_2 = conv2d(layer1_1, 32, 3, 3, 'layer1_2')
            layer1_3 = max_pool(layer1_2, 'layer1_3')           # original image 1/2, (112, 112)
            
            layer2_1 = relu(bn(conv2d(layer1_3, 64, 3, 3, 'layer2_1'), is_training))
            layer2_2 = relu(bn(conv2d(layer2_1, 64, 3, 3, 'layer2_2'), is_training))
            layer2_3 = max_pool(layer2_2, 'layer2_3')           # original image 1/4, (56, 56)

            layer3_1 = relu(bn(conv2d(layer2_3, 128, 3, 3, 'layer3_1'), is_training))
            layer3_2 = relu(bn(conv2d(layer3_1, 128, 3, 3, 'layer3_2'), is_training))
            layer3_3 = max_pool(layer3_2, 'layer3_3')           # original image 1/8, (28, 28)
            
            layer4_1 = relu(bn(conv2d(layer3_3, 256, 3, 3, 'layer4_1'), is_training))
            layer4_2 = relu(bn(conv2d(layer4_1, 256, 3, 3, 'layer4_2'), is_training))
            layer4_3 = max_pool(layer4_2, 'layer4_3')           # original image 1/16, (14, 14)

            layer5_1 = relu(bn(conv2d(layer4_3, 512, 3, 3, 'layer5_1'), is_training))
            layer5_2 = relu(bn(conv2d(layer5_1, 512, 3, 3, 'layer5_2'), is_training))

        with tf.variable_scope('DECODER'):
            layer6_1 = relu(bn(conv2d_t(layer5_2, [self.N_BATCH, 28, 28, 256], 2, 2, 'layer6_1'), is_training))
            layer6_2 = tf.concat([layer4_2, layer6_1], axis=3, name='layer6_2')
            layer6_3 = relu(bn(conv2d(layer6_2, 256, 3, 3, 'layer6_3'), is_training))
            layer6_4 = relu(bn(conv2d(layer6_3, 256, 3, 3, 'layer6_4'), is_training))

            l6_4_shape = layer6_4.get_shape()
            layer7_1 = relu(bn(conv2d_t(layer6_4, [self.N_BATCH, 56, 56, 128], 2, 2, 'layer7_1'), is_training))
            layer7_2 = tf.concat([layer3_2, layer7_1], axis=3, name='layer7_2')
            layer7_3 = relu(bn(conv2d(layer7_2, 128, 3, 3, 'layer7_2'), is_training))
            layer7_4 = relu(bn(conv2d(layer7_3, 128, 3, 3, 'layer7_3'), is_training))

            l7_4_shape = layer7_4.get_shape()
            layer8_1 = relu(bn(conv2d_t(layer7_4, [self.N_BATCH, 112, 112, 64], 2, 2, 'layer8_1'), is_training))
            layer8_2 = tf.concat([layer2_2, layer8_1], axis=3, name='layer8_2')
            layer8_3 = relu(bn(conv2d(layer8_2, 64, 3, 3, 'layer8_3'), is_training))
            layer8_4 = relu(bn(conv2d(layer8_3, 64, 3, 3, 'layer8_4'), is_training))

            l8_4_shape = layer8_4.get_shape()
            layer9_1 = relu(bn(conv2d_t(layer8_4, [self.N_BATCH, 224, 224, 32], 2, 2, 'layer9_1'), is_training))
            layer9_2 = tf.concat([layer1_2, layer9_1], axis=3, name='layer9_2')
            layer9_3 = relu(bn(conv2d(layer9_2, self.N_CLASS, 3, 3, 'layer9_3'), is_training))
            layer9_4 = relu(bn(conv2d(layer9_3, self.N_CLASS, 3, 3, 'layer9_4'), is_training))

            logits = conv2d(layer9_4, self.N_CLASS, 1, 1, 'logits')
            annot_pred = tf.argmax(logits, axis=3)
            expand_pred = tf.expand_dims(annot_pred, dim=3)

            return logits, expand_pred, layer5_2

    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        self.INPUT_Y = tf.placeholder(dtype=tf.int32, shape=[None, 224, 224, 1])

        self.logits, self.pred, _ = self.model(self.INPUT_X)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(self.INPUT_Y, [3]))
        )

        self.optimizer = tf.train.AdamOptimizer(self.L_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('UNET_loss', self.loss)])
        

    def train_model(self):
        data_set_path = load_data_path(self.IMAGE_DATA_DIR, self.ANNOTATION_DATA_DIR, 'training')
        valid_set_path = load_data_path(self.IMAGE_DATA_DIR, self.ANNOTATION_DATA_DIR, 'validation')
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(os.path.join(self.LOG_DIR, self.MODEL_NAME), sess.graph)

            loss_val = 0
            counter = 0

            for epoch in range(self.N_EPOCH):
                random.shuffle(data_set_path)           # 매 epoch마다 데이터셋 shuffling

                for i in range(int(len(data_set_path) / 2)):
                    batch_img_path, batch_ann_path = next_batch(data_set_path, i, self.N_BATCH)
                    batch_imgs, batch_anns = read_image(batch_img_path, batch_ann_path, self.N_BATCH)

                    _, summary_str ,loss_val = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict={self.INPUT_X:batch_imgs, self.INPUT_Y:batch_anns})
                    self.writer.add_summary(summary_str, counter)
                    counter += 1

                print('EPOCH: {}\t'.format(epoch+1), 'LOSS: {:.8}\t'.format(loss_val))

                random.shuffle(valid_set_path)
                valid_img_path, valid_ann_path = next_batch(valid_set_path, epoch, 2)
                valid_img, valid_anns = read_image(valid_img_path, valid_ann_path, 2)

                pred = sess.run(self.pred, feed_dict={self.INPUT_X:valid_img, self.INPUT_Y:valid_anns})[0]
                pred = np.squeeze(pred, axis=2)
                valid_anns = np.squeeze(valid_anns, axis=3)

                # plt.imshow(pred)
                plt.subplot(3, 1, 1)
                plt.imshow(valid_img[0])
                plt.subplot(3, 1, 2)
                plt.imshow(pred)
                plt.subplot(3, 1, 3)
                plt.imshow(valid_anns[0])
                plt.savefig(str(epoch)+'.png')
                
            self.saver.save(sess, './logs/UNET/UNET.ckpt')

    def test_model(self):
        pass
        

    
model = UNET()
model.build_model()
model.train_model()