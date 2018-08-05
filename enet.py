import tensorflow as tf
import numpy as np
import random
import os

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, relu, bn, max_pool, atrous_conv2d
from utils import read_image, load_data_path, next_batch

def prelu(inputs):
    return tf.nn.relu(inputs)
    

def spatial_dropout(inputs, keep_prob, is_training=True):
    if is_training:
        inputs_shape = inputs.get_shape().as_list()
        noise_shape = tf.constant(value=[inputs_shape[0], 1, 1, inputs_shape[3]])
        out = tf.nn.dropout(inputs, keep_prob, noise_shape=noise_shape)
        return out
    
    return inputs


def initial_block(inputs, is_training=True):
    conv = prelu(bn(conv2d(inputs, 13, 3, 3, name='init_conv', strides=[1, 2, 2, 1]), is_training))
    pool = max_pool(inputs, name='init_pool')
    concated = tf.concat([conv, pool], axis=3, name='init_concat')
    return concated

def bottleneck(inputs, out_depth, f_h, f_w, dilated_rate=None, mode=None, scope=None, is_training=True):
    reduce_depth = int(inputs.get_shape().as_list()[3] / 4)
    with tf.variable_scope(scope):
        if mode == 'downsampling':
            main_branch = max_pool(inputs, name='_pool')
            depth_to_pad = abs(inputs.get_shape().as_list()[3] - out_depth)
            paddings = tf.convert_to_tensor([[0,0], [0,0], [0,0], [0, depth_to_pad]])
            main_branch = tf.pad(main_branch, paddings=paddings, name='_main_padding')

            sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 2, 2, name='_conv1', strides=[1, 2, 2, 1]), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, f_h, f_w, name='_conv2', strides=[1, 1, 1, 1]), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, out_depth, 1, 1, name='_conv3', strides=[1, 1, 1, 1]), is_training))
            sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

            out = prelu(tf.add(main_branch, sub_branch))

            return out

        elif mode == 'dilated':
            main_branch = inputs

            sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1', ), is_training))
            sub_branch = prelu(bn(atrous_conv2d(sub_branch, reduce_depth, f_h, f_w, dilated_rate, name='_conv2'), is_training))
            sub_branch = prelu(bn(conv2d(inputs, out_depth, 1, 1, name='_conv3'), is_training))
            sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

            out = prelu(tf.add(main_branch, sub_branch))
            
            return out

        elif mode == 'asymmetric':
            main_branch = inputs
            
            sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1'), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, f_h, 1, name='_conv2'), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, 1, f_w, name='_conv3'), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, out_depth, 1, 1, name='_conv4'), is_training))
            sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

            out = prelu(tf.add(main_branch, sub_branch))
            
            return out

        elif mode == 'upsampling':
            # 논문에서 나오는 unpool 대신 bilinear interpolation 사용
            in_shape = inputs.get_shape().as_list()
            
            main_branch = tf.image.resize_bilinear(inputs, size=[in_shape[1]*2, in_shape[2]*2])
            main_branch = prelu(bn(conv2d(main_branch, out_depth, 3, 3, name='_conv0'), is_training))

            sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1'), is_training))
            sub_branch = prelu(bn(conv2d_t(sub_branch, [in_shape[0], in_shape[1]*2, in_shape[2]*2, reduce_depth], 3, 3, name='_conv2'), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, out_depth, 1, 1, name='_conv3'), is_training))
            sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

            out = prelu(tf.add(main_branch, sub_branch))

            return out
            

        elif mode == 'normal':
            main_branch = inputs

            sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1', strides=[1, 1, 1, 1]), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, f_h, f_w, name='_conv2', strides=[1, 1, 1, 1]), is_training))
            sub_branch = prelu(bn(conv2d(sub_branch, out_depth, 1, 1, name='_conv3', strides=[1, 1, 1, 1]), is_training))
            sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))
            out = prelu(tf.add(main_branch, sub_branch))

            return out



class ENET(object):
    def __init__(self):
        self.N_BATCH = 2
        self.N_EPOCH = 1000
        self.N_CLASS = 151
        self.L_RATE = 1e-5

        self.IMAGE_DATA_DIR = '../dataset/images/'
        self.ANNOTATION_DATA_DIR = '../dataset/annotations/'
        self.LOG_DIR = './logs/ENET/'
        self.MODEL_NAME = 'ENET'


    def model(self, inputs):
        in_shape = inputs.get_shape().as_list()

        with tf.variable_scope('STAGE_INIT'):
            net = initial_block(inputs)
        
        with tf.variable_scope('STAGE_1'):
            net = bottleneck(net, 64, 3, 3, mode='downsampling', scope='bottleneck1.0')
            net = bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.1')
            net = bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.2')
            net = bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.3')
            net = bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.4')

        with tf.variable_scope('STAGE_2'):
            net = bottleneck(net, 128, 3, 3, mode='downsampling', scope='bottleneck2.0')
            net = bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck2.1')
            net = bottleneck(net, 128, 3, 3, dilated_rate=2, mode='dilated', scope='bottleneck2.2')
            net = bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck2.3')
            net = bottleneck(net, 128, 3, 3, dilated_rate=4, mode='dilated', scope='bottleneck2.4')
            net = bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck2.5')
            net = bottleneck(net, 128, 3, 3, dilated_rate=8, mode='dilated', scope='bottleneck2.6')
            net = bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck2.7')
            net = bottleneck(net, 128, 3, 3, dilated_rate=16, mode='dilated', scope='bottleneck2.8')

        with tf.variable_scope('STAGE_3'):
            net = bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck3.0')
            net = bottleneck(net, 128, 3, 3, dilated_rate=2, mode='dilated', scope='bottleneck3.1')
            net = bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck3.2')
            net = bottleneck(net, 128, 3, 3, dilated_rate=4, mode='dilated', scope='bottleneck3.3')
            net = bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck3.4')
            net = bottleneck(net, 128, 3, 3, dilated_rate=8, mode='dilated', scope='bottleneck3.5')
            net = bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck3.6')
            net = bottleneck(net, 128, 3, 3, dilated_rate=16, mode='dilated', scope='bottleneck3.7')

        with tf.variable_scope('STAGE_4'):
            net = bottleneck(net, 64, 3, 3, mode='upsampling', scope='bottleneck4.0')
            net = bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck4.1')
            net = bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck4.2')

        with tf.variable_scope('STAGE_5'):
            net = bottleneck(net, 16, 3, 3, mode='upsampling', scope='bottleneck5.0')
            net = bottleneck(net, 16, 3, 3, mode='normal', scope='bottleneck5.1')

        with tf.variable_scope('STAGE_FULLCONV'):
            net = conv2d_t(net, in_shape[:3]+[64], 2, 2, name='final_conv_t')
            net = conv2d(net, 151, 3, 3, name='final_conv')
            
            return net


    def build_model(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[self.N_BATCH, 224, 224, 3])
        self.INPUT_Y = tf.placeholder(dtype=tf.int32, shape=[self.N_BATCH, 224, 224, 1])

        self.logits = self.model(self.INPUT_X)
        
        self.loss = tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=tf.squeeze(self.INPUT_Y, [3]))

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        
        self.loss_summary = tf.summary.merge([tf.summary.scalar('ENET_loss', self.loss)])

    def train_model(self):
        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)

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
                    batch_imgs, batch_anns = read_image(batch_img_path, batch_ann_path, self.N_BATCH, 224, 224)

                    _, summary_str ,loss_val = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict={self.INPUT_X:batch_imgs, self.INPUT_Y:batch_anns})
                    self.writer.add_summary(summary_str, counter)
                    counter += 1

                print('EPOCH: {}\t'.format(epoch+1), 'LOSS: {:.8}\t'.format(loss_val))


model = ENET()
model.build_model()
model.train_model()