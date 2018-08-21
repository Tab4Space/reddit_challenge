import tensorflow as tf
import numpy as np
from custom_op import conv2d, relu, fully_connect, bn, max_pool, avg_pool, fully_connect
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)

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


class ResNet50(object):
    def __init__(self):
        self.N_EPOCH = 10
        self.N_BATCH = 64
        self.L_RATE = 0.001

        self.MODEL_NAME = 'ResNet50'
        

    def model(self, inputs):
        print('0:', inputs)
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

        x = conv_block(x, [256, 256, 1024], '4_1')
        x = identity_block(x, [256, 256, 1024], '4_2')
        x = identity_block(x, [256, 256, 1024], '4_3')
        x = identity_block(x, [256, 256, 1024], '4_4')
        x = identity_block(x, [256, 256, 1024], '4_5')
        x = identity_block(x, [256, 256, 1024], '4_6')
        print('4: ', x)

        x = conv_block(x, [512, 512, 2048], '5_1')
        x = identity_block(x, [512, 512, 2048], '5_2')
        x = identity_block(x, [512, 512, 2048], '5_3')
        print('5: ', x)

        x = avg_pool(x, 'avg_pool', [1, 7, 7, 1], [1, 1, 1, 1], padding='VALID')
        x = tf.reshape(x, [self.N_BATCH, 2048])
        x = fully_connect(x, 10, 'fc')

        return x

    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.RESIZE_X = tf.image.resize_images(self.INPUT_X, size=[224, 224])
        self.INPUT_Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.logits = self.model(self.RESIZE_X)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.INPUT_Y)

        self.optimizer = tf.train.AdamOptimizer(self.L_RATE).minimize(self.loss)


    def train_model(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            total_batch = int(mnist.train.num_examples / self.N_BATCH)

            for epoch in range(15):
                total_cost = 0

                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(self.N_BATCH)
                    batch_xs = batch_xs.reshape(self.N_BATCH, 28, 28, 1)
                    _, cost_val = sess.run([self.optimizer, self.loss], feed_dict={self.INPUT_X:batch_xs, self.INPUT_Y:batch_ys})
                    total_cost += cost_val

                print('Epoch:', '%04d' % (epoch + 1),'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

        
        
model = ResNet50()
model.build_model()
model.train_model()
    
