import tensorflow as tf
import numpy as np
import os, random, time
import tensorflow.contrib.slim as slim

from tqdm import tqdm
from custom_op import conv2d, conv2d_t, atrous_conv2d, max_pool, bn, prelu, spatial_dropout
from utils import read_data_path, next_batch, read_image, read_annotation, draw_plot


class ENET(object):
    MODEL = 'ENET'

    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate

        self.MODEL_NAME = 'ENET'

        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
        self.N_CLASS = 151
        self.RESIZE = 224
        
        self.TRAIN_IMAGE_PATH = './DATA/ADEChallengeData2016/images/training/'
        self.TRAIN_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/training/'

        self.VALID_IMAGE_PATH = './DATA/ADEChallengeData2016/images/validation/'
        self.VALID_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/validation/'


    def make_model(self, inputs, is_training, keep_prob):
        in_shape = inputs.get_shape().as_list()

        with tf.variable_scope('STAGE_INIT'):
            net = self.initial_block(inputs, is_training)
        
        with tf.variable_scope('STAGE_1'):
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='downsampling', scope='bottleneck1.0')
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck1.1')
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck1.2')
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck1.3')
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck1.4')

        with tf.variable_scope('STAGE_2'):
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, mode='downsampling', scope='bottleneck2.0')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck2.1')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=2, mode='dilated', scope='bottleneck2.2')
            net = self.bottleneck(net, 128, 5, 5, is_training, keep_prob, mode='asymmetric', scope='bottleneck2.3')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=4, mode='dilated', scope='bottleneck2.4')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck2.5')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=8, mode='dilated', scope='bottleneck2.6')
            net = self.bottleneck(net, 128, 5, 5, is_training, keep_prob, mode='asymmetric', scope='bottleneck2.7')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=16, mode='dilated', scope='bottleneck2.8')

        with tf.variable_scope('STAGE_3'):
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck3.0')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=2, mode='dilated', scope='bottleneck3.1')
            net = self.bottleneck(net, 128, 5, 5, is_training, keep_prob, mode='asymmetric', scope='bottleneck3.2')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=4, mode='dilated', scope='bottleneck3.3')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck3.4')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=8, mode='dilated', scope='bottleneck3.5')
            net = self.bottleneck(net, 128, 5, 5, is_training, keep_prob, mode='asymmetric', scope='bottleneck3.6')
            net = self.bottleneck(net, 128, 3, 3, is_training, keep_prob, dilated_rate=16, mode='dilated', scope='bottleneck3.7')

        with tf.variable_scope('STAGE_4'):
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='upsampling', scope='bottleneck4.0')
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck4.1')
            net = self.bottleneck(net, 64, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck4.2')

        with tf.variable_scope('STAGE_5'):
            net = self.bottleneck(net, 16, 3, 3, is_training, keep_prob, mode='upsampling', scope='bottleneck5.0')
            net = self.bottleneck(net, 16, 3, 3, is_training, keep_prob, mode='normal', scope='bottleneck5.1')

        with tf.variable_scope('STAGE_FULLCONV'):
            net = conv2d_t(net, in_shape[:3]+[64], [2, 2], name='final_conv_t')
            pred = conv2d(net, self.N_CLASS, [3, 3], name='pred')
            
            return pred


    def build_model(self):
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.RESIZE, self.RESIZE, 3])         # images
        self.label_y = tf.placeholder(dtype=tf.int32, shape=[None, self.RESIZE, self.RESIZE, 1])         # annotations
        self.is_train = tf.placeholder(dtype=tf.bool)
        self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.pred = self.make_model(self.input_x, self.is_train, self.keep_prob)
        
        """
        첫 번째로, labels_placeholder 에서 나온 값이 32비트 정수로 변환된다.
        그 다음, tf.nn.sparse_softmax_cross_entropy_with_logits가 labels_placeholder에서 1-hot label을 자동으로 생성하고 
        모델의 결과와 비교하여 loss 를 구한다.
        """
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=tf.squeeze(self.label_y, [3])))
        
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])
    
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def train_model(self):
        if not os.path.exists(self.MODEL_NAME+'_result'):   os.mkdir(self.MODEL_NAME+'_result')
        if not os.path.exists(self.LOGS_DIR):   os.path.exists(self.LOGS_DIR)
        if not os.path.exists(self.CKPT_DIR):   os.path.exists(self.CKPT_DIR)
        if not os.path.exists(self.OUTPUT_DIR): os.path.exists(self.OUTPUT_DIR)
        
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

                    feed_dict = {self.input_x: batch_xs, self.label_y: batch_ys, self.is_train: True, self.keep_prob: 0.7}

                    _, summary_str ,loss = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict=feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    counter += 1
                    total_loss += loss

                ## validation 과정
                valid_xs_path, valid_ys_path = next_batch(valid_set_path, epoch, 2)
                valid_xs, valid_ys = read_image(valid_xs_path, valid_ys_path, 2)
                
                feed_dict = {self.input_x: valid_xs, self.label_y: valid_ys, self.is_train:False, self.keep_prob: 1.0}

                valid_pred = sess.run(self.pred, feed_dict=feed_dict)
                valid_pred = np.squeeze(valid_pred, axis=2)
                
                valid_ys = np.squeeze(valid_ys, axis=3)

                ## plotting and save figure
                figure = draw_plot(valid_xs, valid_pred, valid_ys, self.OUTPUT_DIR, epoch, self.batch)
                figure.savefig(self.OUTPUT_DIR + '/' + str(epoch).zfill(3) + '.png')

                print('Epoch:', '%03d' % (epoch + 1), 'Avg Loss: {:.6}\t'.format(total_loss / total_batch))
                self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            
            self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            print('Finish save model')


    def initial_block(self, inputs, is_training):
        conv = prelu(bn(conv2d(inputs, 13, [3, 3], strides=[1, 2, 2, 1], name='init_conv'), is_training), '0')
        pool = max_pool(inputs, name='init_pool')
        concated = tf.concat([conv, pool], axis=3, name='init_concat')
        return concated


    def bottleneck(self, inputs, out_depth, f_h, f_w, is_training, keep_prob, dilated_rate=None, mode=None, scope=None):
        reduce_depth = int(inputs.get_shape().as_list()[3] / 4)
        
        with tf.variable_scope(scope):
            if mode == 'downsampling':
                main_branch = max_pool(inputs, name='_pool')
                depth_to_pad = abs(inputs.get_shape().as_list()[3] - out_depth)
                paddings = tf.convert_to_tensor([[0,0], [0,0], [0,0], [0, depth_to_pad]])
                main_branch = tf.pad(main_branch, paddings=paddings, name='_main_padding')

                sub_branch = prelu(bn(conv2d(inputs, reduce_depth, [2, 2], name='_conv1', strides=[1, 2, 2, 1]), is_training), '1')
                sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, [f_h, f_w], name='_conv2', strides=[1, 1, 1, 1]), is_training), '2')
                sub_branch = prelu(bn(conv2d(sub_branch, out_depth, [1, 1], name='_conv3', strides=[1, 1, 1, 1]), is_training), '3')
                sub_branch = prelu(spatial_dropout(sub_branch, keep_prob), '4')

                out = prelu(tf.add(main_branch, sub_branch), '5')
                return out

            elif mode == 'dilated':
                main_branch = inputs

                sub_branch = prelu(bn(conv2d(inputs, reduce_depth, [1, 1], name='_conv1', ), is_training), '1')
                sub_branch = prelu(bn(atrous_conv2d(sub_branch, reduce_depth, [f_h, f_w], dilated_rate, name='_conv2'), is_training), '2')
                sub_branch = prelu(bn(conv2d(inputs, out_depth, [1, 1], name='_conv3'), is_training), '3')
                sub_branch = prelu(spatial_dropout(sub_branch, keep_prob), '4')

                out = prelu(tf.add(main_branch, sub_branch), '5')
                return out

            elif mode == 'asymmetric':
                main_branch = inputs
                
                sub_branch = prelu(bn(conv2d(inputs, reduce_depth, [1, 1], name='_conv1'), is_training), '1')
                sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, [f_h, 1], name='_conv2'), is_training), '2')
                sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, [1, f_w], name='_conv3'), is_training), '3')
                sub_branch = prelu(bn(conv2d(sub_branch, out_depth, [1, 1], name='_conv4'), is_training), '4')
                sub_branch = prelu(spatial_dropout(sub_branch, keep_prob), '5')

                out = prelu(tf.add(main_branch, sub_branch), '6')
                return out

            elif mode == 'upsampling':
                # 논문에서 나오는 unpool 대신 bilinear interpolation 사용
                in_shape = inputs.get_shape().as_list()
                
                main_branch = tf.image.resize_bilinear(inputs, size=[in_shape[1]*2, in_shape[2]*2])
                main_branch = prelu(bn(conv2d(main_branch, out_depth, [3, 3], name='_conv0'), is_training), '1')

                sub_branch = prelu(bn(conv2d(inputs, reduce_depth, [1, 1], name='_conv1'), is_training), '2')
                sub_branch = prelu(bn(conv2d_t(sub_branch, [in_shape[0], in_shape[1]*2, in_shape[2]*2, reduce_depth], [3, 3], name='_conv2'), is_training), '3')
                sub_branch = prelu(bn(conv2d(sub_branch, out_depth, [1, 1], name='_conv3'), is_training), '4')
                sub_branch = prelu(spatial_dropout(sub_branch, keep_prob), '5')

                out = prelu(tf.add(main_branch, sub_branch), '6')
                return out
                
            elif mode == 'normal':
                main_branch = inputs

                sub_branch = prelu(bn(conv2d(inputs, reduce_depth, [1, 1], name='_conv1', strides=[1, 1, 1, 1]), is_training), '1')
                sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, [f_h, f_w], name='_conv2', strides=[1, 1, 1, 1]), is_training), '2')
                sub_branch = prelu(bn(conv2d(sub_branch, out_depth, [1, 1], name='_conv3', strides=[1, 1, 1, 1]), is_training), '3')
                sub_branch = prelu(spatial_dropout(sub_branch, keep_prob), '4')
                
                out = prelu(tf.add(main_branch, sub_branch), '5')
                return out





# import tensorflow as tf
# import numpy as np
# import os, random
# import tensorflow.contrib.slim as slim

# from tqdm import tqdm
# from custom_op import conv2d, conv2d_t, atrous_conv2d, max_pool, bn, prelu, keep_prob
# from utils import read_data_path, next_batch, read_image, read_annotation, draw_plot


# class ENET(object):
#     def __init__(self, epoch, batch, learning_rate):
#         sself.N_EPOCH = epoch
#         self.N_BATCH = batch
#         self.LEARNING_RATE = learning_rate

#         self.MODEL_NAME = 'FCN8s'

#         self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
#         self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
#         self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')
        
#         self.N_CLASS = 151
#         self.RESIZE = 224
        
#         self.TRAIN_IMAGE_PATH = './DATA/ADEChallengeData2016/images/training/'
#         self.TRAIN_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/training/'

#         self.VALID_IMAGE_PATH = './DATA/ADEChallengeData2016/images/validation/'
#         self.VALID_LABEL_PATH = './DATA/ADEChallengeData2016/annotations/validation/'


#     def make_model(self, inputs, is_traning):
#         in_shape = inputs.get_shape().as_list()

#         with tf.variable_scope('STAGE_INIT'):
#             net = self.initial_block(inputs)
        
#         with tf.variable_scope('STAGE_1'):
#             net = self.bottleneck(net, 64, 3, 3, mode='downsampling', scope='bottleneck1.0')
#             net = self.bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.1')
#             net = self.bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.2')
#             net = self.bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.3')
#             net = self.bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck1.4')

#         with tf.variable_scope('STAGE_2'):
#             net = self.bottleneck(net, 128, 3, 3, mode='downsampling', scope='bottleneck2.0')
#             net = self.bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck2.1')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=2, mode='dilated', scope='bottleneck2.2')
#             net = self.bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck2.3')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=4, mode='dilated', scope='bottleneck2.4')
#             net = self.bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck2.5')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=8, mode='dilated', scope='bottleneck2.6')
#             net = self.bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck2.7')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=16, mode='dilated', scope='bottleneck2.8')

#         with tf.variable_scope('STAGE_3'):
#             net = self.bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck3.0')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=2, mode='dilated', scope='bottleneck3.1')
#             net = self.bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck3.2')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=4, mode='dilated', scope='bottleneck3.3')
#             net = self.bottleneck(net, 128, 3, 3, mode='normal', scope='bottleneck3.4')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=8, mode='dilated', scope='bottleneck3.5')
#             net = self.bottleneck(net, 128, 5, 5, mode='asymmetric', scope='bottleneck3.6')
#             net = self.bottleneck(net, 128, 3, 3, dilated_rate=16, mode='dilated', scope='bottleneck3.7')

#         with tf.variable_scope('STAGE_4'):
#             net = self.bottleneck(net, 64, 3, 3, mode='upsampling', scope='bottleneck4.0')
#             net = self.bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck4.1')
#             net = self.bottleneck(net, 64, 3, 3, mode='normal', scope='bottleneck4.2')

#         with tf.variable_scope('STAGE_5'):
#             net = self.bottleneck(net, 16, 3, 3, mode='upsampling', scope='bottleneck5.0')
#             net = self.bottleneck(net, 16, 3, 3, mode='normal', scope='bottleneck5.1')

#         with tf.variable_scope('STAGE_FULLCONV'):
#             net = conv2d_t(net, in_shape[:3]+[64], [2, 2], name='final_conv_t')
#             net = conv2d(net, 151, [3, 3], name='final_conv')
            
#             return net


#     def build_model(self):
#         self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, self.RESIZE, self.RESIZE, 3])         # images
#         self.label_y = tf.placeholder(dtype=tf.int32, shape=[None, self.RESIZE, self.RESIZE, 1])         # annotations
#         self.is_train = tf.placeholder(dtype=tf.bool)

#         self.logits, self.pred = self.make_model(self.input_x, self.is_train)
        
#         self.loss = tf.reduce_mean(
#             tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.squeeze(self.label_y, [3])))
        
#         self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

#         self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])
    
#         model_vars = tf.trainable_variables()
#         slim.model_analyzer.analyze_vars(model_vars, print_info=True)

#     def train_model(self):
#         if not os.path.exists(self.MODEL_NAME+'_result'):   os.mkdir(self.MODEL_NAME+'_result')
#         if not os.path.exists(self.LOGS_DIR):   os.path.exists(self.LOGS_DIR)
#         if not os.path.exists(self.CKPT_DIR):   os.path.exists(self.CKPT_DIR)
#         if not os.path.exists(self.OUTPUT_DIR): os.path.exists(self.OUTPUT_DIR)
        
#         train_set_path = read_data_path(self.TRAIN_IMAGE_PATH, self.TRAIN_LABEL_PATH)
#         valid_set_path = read_data_path(self.VALID_IMAGE_PATH, self.VALID_LABEL_PATH)

#         with tf.Session() as sess:
#             sess.run(tf.global_variables_initializer())

#             total_batch = int(len(train_set_path) / self.N_BATCH)
#             counter = 0

#             self.saver = tf.train.Saver()
#             self.writer = tf.summary.FileWriter(self.LOGS_DIR, sess.graph)

#             for epoch in range(self.N_EPOCH):
#                 total_loss = 0
#                 random.shuffle(train_set_path)           # 매 epoch마다 데이터셋 shuffling
#                 random.shuffle(valid_set_path)      

#                 for i in range(int(len(train_set_path) / self.N_BATCH)):
#                     batch_xs_path, batch_ys_path = next_batch(train_set_path, self.N_BATCH, i)
#                     batch_xs = read_image(batch_xs_path, [self.RESIZE, self.RESIZE])
#                     batch_ys = read_annotation(batch_ys_path, [self.RESIZE, self.RESIZE])

#                     feed_dict = {self.input_x: batch_xs, self.label_y: batch_ys, self.is_train: True}

#                     _, summary_str ,loss = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict=feed_dict)
#                     self.writer.add_summary(summary_str, counter)
#                     counter += 1
#                     total_loss += loss

#                 ## validation 과정
#                 valid_xs_path, valid_ys_path = next_batch(valid_set_path, epoch, 2)
#                 valid_xs, valid_ys = read_image(valid_xs_path, valid_ys_path, 2)
                
#                 valid_pred = sess.run(self.pred, feed_dict={self.input_x: valid_xs, self.label_y: valid_ys, self.is_train:False})
#                 valid_pred = np.squeeze(validation, axis=2)
                
#                 valid_ys = np.squeeze(valid_ys, axis=3)

#                 ## plotting and save figure
#                 figure = draw_plot(valid_xs, valid_pred, valid_ys, self.OUTPUT_DIR, epoch, self.batch)
#                 figure.savefig(self.OUTPUT_DIR + '/' + str(epoch).zfill(3) + '.png')

#                 print('Epoch:', '%03d' % (epoch + 1), 'Avg Loss: {:.6}\t'.format(total_loss / total_batch))
#                 self.saver.save(self.CKPT_DIR, global_step=counter)

#             self.saver.save(self.CKPT_DIR, global_step=counter)
#             print('Complete save .ckpt file')


#     def initial_block(self, inputs, is_training):
#         conv = prelu(bn(conv2d(inputs, 13, 3, 3, name='init_conv', strides=[1, 2, 2, 1]), is_training))
#         pool = max_pool(inputs, name='init_pool')
#         concated = tf.concat([conv, pool], axis=3, name='init_concat')
#         return concated


#     def bottleneck(self, inputs, out_depth, f_h, f_w, is_training, dilated_rate=None, mode=None, scope=None):
#         reduce_depth = int(inputs.get_shape().as_list()[3] / 4)
        
#         with tf.variable_scope(scope):
#             if mode == 'downsampling':
#                 main_branch = max_pool(inputs, name='_pool')
#                 depth_to_pad = abs(inputs.get_shape().as_list()[3] - out_depth)
#                 paddings = tf.convert_to_tensor([[0,0], [0,0], [0,0], [0, depth_to_pad]])
#                 main_branch = tf.pad(main_branch, paddings=paddings, name='_main_padding')

#                 sub_branch = prelu(bn(conv2d(inputs, reduce_depth, [2, 2], name='_conv1', strides=[1, 2, 2, 1]), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, f_h, f_w, name='_conv2', strides=[1, 1, 1, 1]), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, out_depth, [1, 1], name='_conv3', strides=[1, 1, 1, 1]), is_training))
#                 sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

#                 out = prelu(tf.add(main_branch, sub_branch))
#                 return out

#             elif mode == 'dilated':
#                 main_branch = inputs

#                 sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1', ), is_training))
#                 sub_branch = prelu(bn(atrous_conv2d(sub_branch, reduce_depth, f_h, f_w, dilated_rate, name='_conv2'), is_training))
#                 sub_branch = prelu(bn(conv2d(inputs, out_depth, 1, 1, name='_conv3'), is_training))
#                 sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

#                 out = prelu(tf.add(main_branch, sub_branch))
#                 return out

#             elif mode == 'asymmetric':
#                 main_branch = inputs
                
#                 sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1'), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, f_h, 1, name='_conv2'), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, 1, f_w, name='_conv3'), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, out_depth, 1, 1, name='_conv4'), is_training))
#                 sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

#                 out = prelu(tf.add(main_branch, sub_branch))
#                 return out

#             elif mode == 'upsampling':
#                 # 논문에서 나오는 unpool 대신 bilinear interpolation 사용
#                 in_shape = inputs.get_shape().as_list()
                
#                 main_branch = tf.image.resize_bilinear(inputs, size=[in_shape[1]*2, in_shape[2]*2])
#                 main_branch = prelu(bn(conv2d(main_branch, out_depth, 3, 3, name='_conv0'), is_training))

#                 sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1'), is_training))
#                 sub_branch = prelu(bn(conv2d_t(sub_branch, [in_shape[0], in_shape[1]*2, in_shape[2]*2, reduce_depth], 3, 3, name='_conv2'), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, out_depth, 1, 1, name='_conv3'), is_training))
#                 sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))

#                 out = prelu(tf.add(main_branch, sub_branch))
#                 return out
                
#             elif mode == 'normal':
#                 main_branch = inputs

#                 sub_branch = prelu(bn(conv2d(inputs, reduce_depth, 1, 1, name='_conv1', strides=[1, 1, 1, 1]), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, reduce_depth, f_h, f_w, name='_conv2', strides=[1, 1, 1, 1]), is_training))
#                 sub_branch = prelu(bn(conv2d(sub_branch, out_depth, 1, 1, name='_conv3', strides=[1, 1, 1, 1]), is_training))
#                 sub_branch = prelu(spatial_dropout(sub_branch, 0.7, is_training))
                
#                 out = prelu(tf.add(main_branch, sub_branch))
#                 return out
