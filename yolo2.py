import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow.contrib.slim as slim

from custom_op import conv2d, max_pool, avg_pool, lrelu, bn, calc_iou
from utils import next_batch, read_image, read_xml, read_data_path


class YOLO_V2(object):
    MODEL = 'YOLO_v2'
    
    def __init__(self, epoch, batch, learning_rate):
        self.N_EPOCH = epoch
        self.N_BATCH = batch
        self.LEARNING_RATE = learning_rate
        
        self.N_CLASSES = 20
        self.N_ANCHORS = 5
        self.GRID_SHAPE = [13, 13]
        self.IMAGE_SHAPE = [416, 416, 3]
        self.LAMBDA_COORD = 5
        self.LAMBDA_OBJ = 0.5
        self.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        
        self.ANCHOR = [0.57273, 1.87446, 3.33843, 7.88282, 9.77052, 0.677385, 2.06253, 5.47434, 3.52778, 9.16828]

        self.offset = np.transpose(np.reshape(np.array([np.arange(self.GRID_SHAPE[0])] * self.GRID_SHAPE[0] * self.N_ANCHORS),
                                         [self.N_ANCHORS, self.GRID_SHAPE[0], self.GRID_SHAPE[0]]), (1, 2, 0))
        self.offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32), [1, self.GRID_SHAPE[0], self.GRID_SHAPE[0], self.N_ANCHORS])
        self.offset = tf.tile(self.offset, (self.N_BATCH, 1, 1, 1))

        self.MODEL_NAME = 'YOLO_v2'
        self.LOGS_DIR = os.path.join(self.MODEL_NAME+'_result', 'logs')
        self.CKPT_DIR = os.path.join(self.MODEL_NAME+'_result', 'ckpt')
        self.OUTPUT_DIR = os.path.join(self.MODEL_NAME+'_result', 'output')

        self.TRAIN_IMAGE_PATH = './DATA/PASCAL_VOC_2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
        self.TRAIN_ANNOT_PATH = './DATA/PASCAL_VOC_2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'


    def make_model(self, inputs, is_training):
        with tf.variable_scope('Darknet19'):
            net = lrelu(bn(conv2d(inputs, 32, [3, 3], name='conv1'), is_training))
            net = max_pool(net, name='pool1')

            net = lrelu(bn(conv2d(net, 64, [3, 3], name='conv2'), is_training))
            net = max_pool(net, name='pool2')

            net = lrelu(bn(conv2d(net, 128, [3, 3], name='conv3'), is_training))
            net = lrelu(bn(conv2d(net, 64, [1, 1], name='conv4'), is_training))
            net = lrelu(bn(conv2d(net, 128, [3, 3], name='conv5'), is_training))
            net = max_pool(net, name='pool3')

            net = lrelu(bn(conv2d(net, 256, [3, 3], name='conv6'), is_training))
            net = lrelu(bn(conv2d(net, 128, [1, 1], name='conv7'), is_training))
            net = lrelu(bn(conv2d(net, 256, [3, 3], name='conv8'), is_training))
            net = max_pool(net, name='pool4')

            net = lrelu(bn(conv2d(net, 512, [3, 3], name='conv9'), is_training))
            net = lrelu(bn(conv2d(net, 256, [1, 1], name='conv10'), is_training))
            net = lrelu(bn(conv2d(net, 512, [3, 3], name='conv11'), is_training))
            net = lrelu(bn(conv2d(net, 256, [3, 3], name='conv12'), is_training))
            skip = lrelu(bn(conv2d(net, 512, [3, 3], name='conv13'), is_training))
            net = max_pool(net, name='pool5')

            net = lrelu(bn(conv2d(net, 1024, [3, 3], name='conv14'), is_training))
            net = lrelu(bn(conv2d(net, 512, [1, 1], name='conv15'), is_training))
            net = lrelu(bn(conv2d(net, 1024, [3, 3], name='conv16'), is_training))
            net = lrelu(bn(conv2d(net, 512, [1, 1], name='conv17'), is_training))
            net = lrelu(bn(conv2d(net, 1024, [3, 3], name='conv18'), is_training))

        with tf.variable_scope('Detection'):
            net = lrelu(bn(conv2d(net, 1024, [3, 3], name='conv19'), is_training))
            net = lrelu(bn(conv2d(net, 1024, [3, 3], name='conv20'), is_training))

            passthrough = lrelu(bn(conv2d(skip, 64, [1, 1], name='conv21_passthrough'), is_training))
            passthrough = tf.space_to_depth(passthrough, block_size=2)

            concated = tf.concat([passthrough, net], axis=3)

            net = lrelu(bn(conv2d(concated, 1024, [3, 3], name='conv22'), is_training))

            out_depth = self.N_ANCHORS * (5 + self.N_CLASSES)
            net = conv2d(net, out_depth, [1, 1], name='conv23')
            
            return net

            
    def loss_layer(self, pred, label):
        pred = tf.reshape(pred, [-1, 13, 13, self.N_ANCHORS, 5+self.N_CLASSES])
        pred_box_coordinate = tf.reshape(pred[..., :4], [-1, 13, 13, self.N_ANCHORS, 4])
        pred_box_confidence = tf.reshape(pred[..., 4], [-1, 13, 13, self.N_ANCHORS, 1])
        pred_box_classes = tf.reshape(pred[..., 5:], [-1, 13, 13, self.N_ANCHORS, self.N_CLASSES])

        # Tensor("stack:0", shape=(4, 8, 13, 13, 5), dtype=float32) 
        box = tf.stack([(tf.nn.sigmoid(pred_box_coordinate[..., 0]) + self.offset) / self.GRID_SHAPE[0],                                   # x 좌표만
                        (tf.nn.sigmoid(pred_box_coordinate[..., 1]) + tf.transpose(self.offset, [0, 2, 1, 3]) / self.GRID_SHAPE[0]),       # y 좌표만
                        tf.sqrt(tf.exp(pred_box_coordinate[..., 2]) * np.reshape(self.ANCHOR[:5], [1, 1, 1, 5]) / self.GRID_SHAPE[0]),     # w 너비만
                        tf.sqrt(tf.exp(pred_box_coordinate[..., 3]) * np.reshape(self.ANCHOR[5:], [1, 1, 1, 5]) / self.GRID_SHAPE[0])      # h 높이만
                        ])
        
        # Tensor("transpose_1:0", shape=(8, 13, 13, 5, 4), dtype=float32)
        box_T = tf.transpose(box, [1, 2, 3, 4, 0])
        box_conf = tf.sigmoid(pred_box_confidence)
        box_class = tf.nn.softmax(pred_box_classes)

        label_box_coordinate = tf.reshape(label[..., 1:5], [-1, 13, 13, self.N_ANCHORS, 4])
        label_box_confidence = tf.reshape(label[..., 0], [-1, 13, 13, self.N_ANCHORS])
        label_box_classes = tf.reshape(label[..., 5:], [-1, 13, 13, self.N_ANCHORS, self.N_CLASSES])

        # print('box_T: ', box_T)
        # print('label_box_coordinate: ', label_box_coordinate)

        iou = calc_iou(box_T, label_box_coordinate)
        best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))
        confs = tf.expand_dims(best_box * label_box_confidence, axis=4)

        conf_id = 1.0 * (1.0 - confs) + 5.0 * confs
        coor_id = 1.0 * confs
        prob_id = 1.0 * confs

        conf_loss = conf_id * tf.square(box_conf - confs)
        coor_loss = coor_id * tf.square(box_T - label_box_coordinate)
        prob_loss = prob_id * tf.square(box_class - label_box_classes)

        loss = tf.concat([coor_loss, conf_loss, prob_loss], axis=4)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]))

        return loss

    
    def build_model(self):
        self.input_x = tf.placeholder(tf.float32, shape=[None]+self.IMAGE_SHAPE)
        self.label_y = tf.placeholder(tf.float32, shape=[None]+self.GRID_SHAPE+[self.N_ANCHORS, 5+self.N_CLASSES])
        self.is_train = tf.placeholder(tf.bool)

        self.pred = self.make_model(self.input_x, self.is_train)
        self.loss = self.loss_layer(self.pred, self.label_y)

        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

        self.loss_summary = tf.summary.merge([tf.summary.scalar('loss', self.loss)])
    
        # model_vars = tf.trainable_variables()
        # slim.model_analyzer.analyze_vars(model_vars, print_info=True)
        

    def train_model(self):
        if not os.path.exists(self.MODEL_NAME+'_result'):   os.mkdir(self.MODEL_NAME+'_result')
        if not os.path.exists(self.LOGS_DIR):   os.mkdir(self.LOGS_DIR)
        if not os.path.exists(self.CKPT_DIR):   os.mkdir(self.CKPT_DIR)
        if not os.path.exists(self.OUTPUT_DIR): os.mkdir(self.OUTPUT_DIR)

        train_set_path = read_data_path(self.TRAIN_IMAGE_PATH, self.TRAIN_ANNOT_PATH)

        ckpt_save_path = os.path.join(self.CKPT_DIR, self.MODEL_NAME+'_'+str(self.N_BATCH)+'_'+str(self.LEARNING_RATE))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            total_batch = int(len(train_set_path) / self.N_BATCH)
            counter = 0

            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.LOGS_DIR, sess.graph)

            for epoch in range(self.N_EPOCH):
                total_loss = 0
                
                for i in range(int(len(train_set_path) / self.N_BATCH)):
                    batch_xs_path, batch_ys_path = next_batch(train_set_path, self.N_BATCH, i)
                    batch_xs = read_image(batch_xs_path, self.IMAGE_SHAPE[:2])
                    batch_ys = read_xml(batch_ys_path, self.N_BATCH, self.IMAGE_SHAPE[0], self.GRID_SHAPE[0], self.N_ANCHORS, self.N_CLASSES, self.CLASSES)

                    feed_dict = {self.input_x: batch_xs, self.label_y:batch_ys, self.is_train: True}
                    _, summary_str, loss = sess.run([self.optimizer, self.loss_summary, self.loss], feed_dict=feed_dict)
                
                    self.writer.add_summary(summary_str, counter)
                    total_loss += loss
                    counter += 1

                print('Epoch:', '%03d' % (epoch + 1), 'Avg Loss: {:.6}\t'.format(total_loss / total_batch))
                self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            
            self.saver.save(sess, ckpt_save_path+'_'+str(epoch)+'.model', global_step=counter)
            print('Finish save model')

