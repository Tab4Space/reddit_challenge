import tensorflow as tf
import numpy as np
from tqdm import tqdm
from custom_op import conv2d, conv2d_t, max_pool, lrelu, bn
from utils import load_data_path, next_batch, read_image
import scipy.misc
import random

class FCN32(object):
    def __init__(self):
        self.N_BATCH = 2
        self.N_EPOCH = 100000
        self.N_CLASS = 151
        self.L_RATE = 1e-5

        self.IMAGE_DATA_DIR = '../dataset/images/'
        self.ANNOTATION_DATA_DIR = '../dataset/annotations/'

    def model(self, inputs, is_training=True):
        inputs_shape = inputs.get_shape()

        layer1_1 = lrelu(conv2d(inputs, 64, 3, 3, 'layer1_1'))
        layer1_2 = lrelu(conv2d(layer1_1, 64, 3, 3, 'layer1_2'))
        layer1_3 = max_pool(layer1_2, 'layer1_3')       # original image 1/2
        
        layer2_1 = lrelu(bn(conv2d(layer1_3, 128, 3, 3, 'layer2_1'), is_training))
        layer2_2 = lrelu(bn(conv2d(layer2_1, 128, 3, 3, 'layer2_2'), is_training))
        layer2_3 = max_pool(layer2_2, 'layer2_3')       # original image 1/4

        layer3_1 = lrelu(bn(conv2d(layer2_3, 256, 3, 3, 'layer3_1'), is_training))
        layer3_2 = lrelu(bn(conv2d(layer3_1, 256, 3, 3, 'layer3_2'), is_training))
        layer3_3 = max_pool(layer3_2, 'layer3_3')       # original image 1/8
        
        layer4_1 = lrelu(bn(conv2d(layer3_3, 512, 3, 3, 'layer4_1'), is_training))
        layer4_2 = lrelu(bn(conv2d(layer4_1, 512, 3, 3, 'layer4_2'), is_training))
        layer4_3 = max_pool(layer4_2, 'layer4_3')       # original image 1/16

        layer5_1 = lrelu(bn(conv2d(layer4_3, 512, 3, 3, 'layer5_1'), is_training))
        layer5_2 = lrelu(bn(conv2d(layer5_1, 512, 3, 3, 'layer5_2'), is_training))
        layer5_3 = max_pool(layer5_2, 'layer5_3')       # original image 1/32

        l5_3_shape = layer5_3.get_shape()
        # make [batch, 1, 1, 2048] similary flatten in fully connected layer
        layer6_1 = lrelu(bn(conv2d(layer5_3, 2048, 7, 7, 'layer6_1'), is_training))
        layer6_2 = lrelu(bn(conv2d(layer6_1, 2048, 1, 1, 'layer6_2'), is_training))
        layer6_3 = lrelu(bn(conv2d(layer6_2, self.N_CLASS, 1, 1, 'layer6_3'), is_training))
        print(layer6_3.get_shape())


        # FCN32 is not use previous pooling information
        # just last layer size up(x32)
        # conv2d_transpose로 upscaling 할때, strides 크기로 결정됨.
        # 만약, 32배로 사이즈를 늘리려면 strides=[1, 32, 32, 1], 16배로 늘리려면 strides=[1, 16, 16, 1]로 하면 되는 듯하다.
        layer7_1 = conv2d_t(layer6_3, [self.N_BATCH, 224, 224, self.N_CLASS], 4, 4, 'layer7_1', strides=[1, 2, 2, 1])
        layer7_2 = tf.argmax(layer7_1, axis=3)
        return layer7_2

    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])         # images
        self.INPUT_Y = tf.placeholder(dtype=tf.int32, shape=[None, 224, 224, 1])         # annotations

        logits = self.model(self.INPUT_X)

        
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(self.INPUT_Y, [3])))
        
        self.optimizer = tf.train.AdamOptimizer(self.L_RATE).minimize(self.loss)

    
    def train_model(self):
        data_set_path = load_data_path(self.IMAGE_DATA_DIR, self.ANNOTATION_DATA_DIR, 'training')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_val = 0

            for epoch in range(self.N_EPOCH):
                random.shuffle(data_set_path)           # 매 epoch마다 데이터셋 shuffling

                for i in range(int(len(data_set_path) / 2)):
                    batch_img_path, batch_ann_path = next_batch(data_set_path, i, self.N_BATCH)
                    batch_imgs, batch_anns = read_image(batch_img_path, batch_ann_path, self.N_BATCH)

                    _, loss_val = sess.run([self.optimizer, self.loss], feed_dict={self.INPUT_X:batch_imgs, self.INPUT_Y:batch_anns})

                print('EPOCH: {}\t'.format(epoch+1), 'LOSS: {:.8}\t'.format(loss_val))

    def test_model(self):
        pass


model = FCN32()
model.build_model()
model.train_model()