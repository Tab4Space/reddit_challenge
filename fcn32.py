import tensorflow as tf
import numpy as np
from tqdm import tqdm
from custom_op import conv2d, conv2d_t, max_pool, relu, bn, lrelu
from utils import load_data_path, next_batch, read_image
import scipy.misc
import random
import os
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)

class FCN32(object):
    def __init__(self):
        self.N_BATCH = 2
        self.N_EPOCH = 100000
        self.N_CLASS = 151
        self.L_RATE = 1e-5

        self.IMAGE_DATA_DIR = '../dataset/images/'
        self.ANNOTATION_DATA_DIR = '../dataset/annotations/'
        self.LOGS_DIR = './logs/'

    def model(self, inputs, is_training=True):
        layer1_1 = relu(conv2d(inputs, 64, 3, 3, 'layer1_1'))
        print('1_1: ', layer1_1)
        layer1_2 = relu(conv2d(layer1_1, 64, 3, 3, 'layer1_2'))
        print('1_2: ', layer1_2)
        layer1_3 = max_pool(layer1_2, 'layer1_3')       # original image 1/2
        print('1_3: ', layer1_3)
        
        layer2_1 = relu(bn(conv2d(layer1_3, 128, 3, 3, 'layer2_1'), is_training))
        print('2_1: ', layer2_1)
        layer2_2 = relu(bn(conv2d(layer2_1, 128, 3, 3, 'layer2_2'), is_training))
        print('2_2: ', layer2_2)
        layer2_3 = max_pool(layer2_2, 'layer2_3')       # original image 1/4
        print('2_3: ', layer2_3)

        layer3_1 = relu(bn(conv2d(layer2_3, 256, 3, 3, 'layer3_1'), is_training))
        print('3_1: ', layer3_1)
        layer3_2 = relu(bn(conv2d(layer3_1, 256, 3, 3, 'layer3_2'), is_training))
        print('3_2: ', layer3_2)
        layer3_3 = max_pool(layer3_2, 'layer3_3')       # original image 1/8
        print('3_3: ', layer3_3)
        
        layer4_1 = relu(bn(conv2d(layer3_3, 512, 3, 3, 'layer4_1'), is_training))
        print('4_1: ', layer4_1)
        layer4_2 = relu(bn(conv2d(layer4_1, 512, 3, 3, 'layer4_2'), is_training))
        print('4_2: ', layer4_2)
        layer4_3 = max_pool(layer4_2, 'layer4_3')       # original image 1/16
        print('4_3: ', layer4_3)

        layer5_1 = relu(bn(conv2d(layer4_3, 512, 3, 3, 'layer5_1'), is_training))
        print('5_1: ', layer5_1)
        layer5_2 = relu(bn(conv2d(layer5_1, 512, 3, 3, 'layer5_2'), is_training))
        print('5_2: ', layer5_2)
        layer5_3 = max_pool(layer5_2, 'layer5_3')       # original image 1/32
        print('5_3: ', layer5_3)

        l5_3_shape = layer5_3.get_shape()
        # make [batch, 1, 1, 2048] similary flatten in fully connected layer
        layer6_1 = relu(bn(conv2d(layer5_3, 2048, 7, 7, 'layer6_1'), is_training))
        print('6_1: ', layer6_1)
        layer6_2 = relu(bn(conv2d(layer6_1, 2048, 1, 1, 'layer6_2'), is_training))
        print('6_2: ', layer6_2)
        layer6_3 = relu(bn(conv2d(layer6_2, self.N_CLASS, 1, 1, 'layer6_3'), is_training))
        print('6_3: ', layer6_3)


        # FCN32 is not use previous pooling information
        # just last layer size up(x32)
        # conv2d_transpose로 upscaling 할때, strides 크기로 결정됨.
        # 만약, 32배로 사이즈를 늘리려면 strides=[1, 32, 32, 1], 16배로 늘리려면 strides=[1, 16, 16, 1]로 하면 되는 듯하다.
        layer7_1 = conv2d_t(layer6_3, [self.N_BATCH, 224, 224, self.N_CLASS], 4, 4, 'layer7_1', strides=[1, 32, 32, 1])
        print('7_1: ', layer7_1)

        annot_pred = tf.argmax(layer7_1, axis=3)
        print('annot_pred: ', annot_pred)

        expand_pred = tf.expand_dims(annot_pred, dim=3)
        print('expand_pred: ', expand_pred)

        return layer7_1, expand_pred

    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])         # images
        self.INPUT_Y = tf.placeholder(dtype=tf.int32, shape=[None, 224, 224, 1])         # annotations

        self.logits, self.pred = self.model(self.INPUT_X)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(self.INPUT_Y, [3])))
        
        self.optimizer = tf.train.AdamOptimizer(self.L_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])


    
    def train_model(self):
        data_set_path = load_data_path(self.IMAGE_DATA_DIR, self.ANNOTATION_DATA_DIR, 'training')
        valid_set_path = load_data_path('../dataset/images/', '../dataset/annotations/', 'validation')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(os.path.join(self.LOGS_DIR, 'FCN32'), sess.graph)

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
                print('pred1: ', pred.shape)

                pred = np.squeeze(pred, axis=2)
                print('pred2: ', pred.shape)

                print(pred)

                plt.imshow(pred)
                
                plt.imsave('aaa.png', pred)
                plt.show()


            self.saver.save(sess, './logs/FCN32.ckpt')

    def test_model(self):
        pass


model = FCN32()
model.build_model()
model.train_model()