import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.misc as misc

from custom_op import conv2d, max_pool, avg_pool, lrelu, bn
from yolo_utils import read_yolo_data_path, yolo_next_batch, yolo_read_image, yolo_read_annot

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

ANCHOR = [0.57273, 1.87446, 3.33843, 7.88282, 9.77052, 0.677385, 2.06253, 5.47434, 3.52778, 9.16828]

def calc_iou(boxes1, boxes2):
        boxx = tf.square(boxes1[:, :, :, :, 2:4])
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        boxx = tf.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))

        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(right_down - left_up, 0.0)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        union_square = boxes1_square + boxes2_square - inter_square

        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)

def test(logits, path):
    
    def calc_output(output):
        output = np.reshape(output, [13, 13, 5, 5 + 20])
        
        # (13, 13, 5, 25-> 앞에 4개(0~3)는 box의 좌표정보)
        # boxes의 전체 element 갯수는 3380

        boxes = np.reshape(output[:, :, :, :4], [13, 13, 5, 4])    #boxes coordinate
        boxes = get_boxes(boxes) * 416
        
        # (13, 13, 5, 25-> 4번째 위치한 element는 confidence 정보)
        confidence = np.reshape(output[:, :, :, 4], [13, 13, 5])    #the confidence of the each anchor boxes
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence))
        confidence = np.tile(np.expand_dims(confidence, 3), (1, 1, 1, 20))

        # (13, 13, 5, 25-> 5번째부터 끝까지는 class 정보)
        classes = np.reshape(output[:, :, :, 5:], [13, 13, 5, 20])    #classes
        classes = np.exp(classes) / np.tile(np.expand_dims(np.sum(np.exp(classes), axis=3), axis=3), (1, 1, 1, 20))

        probs = classes * confidence

        filter_probs = np.array(probs >= 0.3, dtype = 'bool')
        filter_index = np.nonzero(filter_probs)
        box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]]
        probs_filter = probs[filter_probs]
        classes_num = np.argmax(filter_probs, axis = 3)[filter_index[0], filter_index[1], filter_index[2]]

        # 가장 신뢰도가 높은 bbox 순서로 정렬하는 듯
        sort_num = np.array(np.argsort(probs_filter))[::-1]
        box_filter = box_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        classes_num = classes_num[sort_num]

        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                if iou(box_filter[i], box_filter[j]) > 0.5:
                    probs_filter[j] = 0.0

        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        probs_filter = probs_filter[filter_probs]
        box_filter = box_filter[filter_probs]
        classes_num = classes_num[filter_probs]

        results = []
        for i in range(len(probs_filter)):
            results.append([CLASSES[classes_num[i]], box_filter[i][0], box_filter[i][1],
                            box_filter[i][2], box_filter[i][3], probs_filter[i]])

        # print('finish calc_output')
        # print(results)
        return results

    def iou(box1, box2):
        width = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        height = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height

        #print('finish calc_iou')
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def get_boxes(self, boxes):
        """
        실제 테스트 이미지(416, 416) 상에서 표현될 box를 구하는 듯
        모델에서 나온 feature map 상에서의 정보를 실제 이미지 상에서 표현하기 위한 작업
        """
        #print('run get_boxes')
        offset = np.transpose(np.reshape(np.array([np.arange(13)] * 13 * 5),
                                         [5, 13, 13]), (1, 2, 0))
        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 0])) + offset) / 13,
                           (1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 1])) + np.transpose(offset, (1, 0, 2))) / 13,
                           np.exp(boxes[:, :, :, 2]) * np.reshape(ANCHOR[:5], [1, 1, 5]) / 13,
                           np.exp(boxes[:, :, :, 3]) * np.reshape(ANCHOR[5:], [1, 1, 5]) / 13])

        #print('finish get_boxes')
        return np.transpose(boxes1, (1, 2, 3, 0))

    def random_colors(self, N, bright=True):
        #print('run random_colors')
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)

        #print('finish random_colors')
        return colors


    def draw(self, image, result):
        #print('run draw')
        image_h, image_w, _ = image.shape
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, result[i][0] + ':%.2f' % result[i][5], (xmin + 1, ymin + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)



    image = misc.imread(path)
    results = iou(logits)
    draw(image, results)
    misc.imshow(image)

class YOLO_V2(object):
    def __init__(self):
        self.N_EPOCH = 50
        self.N_BATCH = 16
        self.L_RATE = 0.00001

        self.N_CLASSES = 20
        self.N_ANCHORS = 5
        self.GRID_H, self.GRID_W = 13, 13
        self.IMG_H, self.IMG_W, self.IMG_C = 416, 416, 3
        self.LAMBDA_COORD = 5
        self.LAMBDA_OBJ = 0.5
        self.ANCHOR = [0.57273, 1.87446, 3.33843, 7.88282, 9.77052, 0.677385, 2.06253, 5.47434, 3.52778, 9.16828]

        self.offset = np.transpose(np.reshape(np.array([np.arange(self.GRID_H)] * self.GRID_H * self.N_ANCHORS),
                                         [self.N_ANCHORS, self.GRID_H, self.GRID_H]), (1, 2, 0))
        self.offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32), [1, self.GRID_H, self.GRID_H, self.N_ANCHORS])
        self.offset = tf.tile(self.offset, (self.N_BATCH, 1, 1, 1))

        self.TRAIN_DATA_IMG_PATH = 'd:/_PASCAL_VOC_2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/'
        self.TRAIN_DATA_ANN_PATH = 'd:/_PASCAL_VOC_2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'

        
    def model(self, inputs, is_training=True):
        with tf.variable_scope('Darknet19'):
            # conv1 - bn - leaky_relu - pool1
            # (batch, 416, 416, 3) -> (batch, 208, 208, 32)
            net = lrelu(bn(conv2d(inputs, 32, 3, 3, name='conv1'), is_training))
            net = max_pool(net, name='pool1')

            # conv2 - bn - leaky_relu - pool2
            # (batch, 208, 208, 32) -> (batch, 104, 104, 64)
            net = lrelu(bn(conv2d(net, 64, 3, 3, name='conv2'), is_training))
            net = max_pool(net, name='pool2')

            # conv3 - bn - leaky_relu
            # conv4 - bn - leaky_relu
            # conv5 - bn - leaky_relu - pool3
            # (batch, 104, 104, 64) -> (batch, 52, 52, 128)
            net = lrelu(bn(conv2d(net, 128, 3, 3, name='conv3'), is_training))
            net = lrelu(bn(conv2d(net, 64, 1, 1, name='conv4'), is_training))
            net = lrelu(bn(conv2d(net, 128, 3, 3, name='conv5'), is_training))
            net = max_pool(net, name='pool3')

            # conv6 - bn - leaky_relu
            # conv7 - bn - leaky_relu
            # conv8 - bn - leaky_relu - pool4
            # (batch, 52, 52, 128) -> (batch, 26, 26, 256)
            net = lrelu(bn(conv2d(net, 256, 3, 3, name='conv6'), is_training))
            net = lrelu(bn(conv2d(net, 128, 1, 1, name='conv7'), is_training))
            net = lrelu(bn(conv2d(net, 256, 3, 3, name='conv8'), is_training))
            net = max_pool(net, name='pool4')

            # conv9 - bn - leaky_relu
            # conv10 - bn - leaky_relu
            # conv11 - bn - leaky_relu
            # conv12 - bn - leaky_relu
            # conv13 - bn - leaky_relu - pool5
            # (batch, 26, 26, 256) -> (batch, 13, 13, 512)
            net = lrelu(bn(conv2d(net, 512, 3, 3, name='conv9'), is_training))
            net = lrelu(bn(conv2d(net, 256, 1, 1, name='conv10'), is_training))
            net = lrelu(bn(conv2d(net, 512, 3, 3, name='conv11'), is_training))
            net = lrelu(bn(conv2d(net, 256, 1, 1, name='conv12'), is_training))
            skip = lrelu(bn(conv2d(net, 512, 3, 3, name='conv13'), is_training))
            net = max_pool(skip, name='pool5')

            # conv14 - bn - leaky_relu
            # conv15 - bn - leaky_relu
            # conv16 - bn - leaky_relu
            # conv17 - bn - leaky_relu
            # conv18 - bn - leaky_relu
            # (batch, 13, 13, 512) -> (batch, 13, 13, 1024)
            net = lrelu(bn(conv2d(net, 1024, 3, 3, name='conv14'), is_training))
            net = lrelu(bn(conv2d(net, 512, 1, 1, name='conv15'), is_training))
            net = lrelu(bn(conv2d(net, 1024, 3, 3, name='conv16'), is_training))
            net = lrelu(bn(conv2d(net, 512, 1, 1, name='conv17'), is_training))
            net = lrelu(bn(conv2d(net, 1024, 3, 3, name='conv18'), is_training))


        with tf.variable_scope('detection'):
            # (batch, 13, 13, 1024)
            net = lrelu(bn(conv2d(net, 1024, 3, 3, name='conv19'), is_training))
            
            # (batch, 13, 13, 1024)
            net = lrelu(bn(conv2d(net, 1024, 3, 3, name='conv20'), is_training))

            # passthrough layer
            # (batch, 13, 13, 256 + 1024)
            passthrough = lrelu(bn(conv2d(skip, 64, 1, 1, name='conv21_passthrough'), is_training))
            passthrough = tf.space_to_depth(passthrough, block_size=2)
            concated = tf.concat([passthrough, net], axis=3)

            # conv22 - bn - leaky_relu
            # (batch, 13, 13, 1280) -> (batch, 13, 13, 1024)
            net = lrelu(bn(conv2d(concated, 1024, 3, 3, name='conv22'), is_training))

            # 
            output_channel = self.N_ANCHORS * (5 + self.N_CLASSES)
            net = conv2d(net, output_channel, 1, 1, name='conv23')
            return net


    def loss_layer(self, pred, label):
        # label 만들때, [conf, x, y, w, h, class] 순서임
        pred = tf.reshape(pred, [-1, 13, 13, self.N_ANCHORS, 5+self.N_CLASSES])
        pred_box_coordinate = tf.reshape(pred[..., :4], [-1, 13, 13, self.N_ANCHORS, 4])
        pred_box_confidence = tf.reshape(pred[..., 4], [-1, 13, 13, self.N_ANCHORS, 1])
        pred_box_classes = tf.reshape(pred[..., 5:], [-1, 13, 13, self.N_ANCHORS, self.N_CLASSES])


        # boxes1 = tf.stack([(1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 0])) + self.offset) / self.cell_size,
        #                    (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + tf.transpose(self.offset, (0, 2, 1, 3))) / self.cell_size,
        #                    tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 1, 5]) / self.cell_size),
        #                    tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 1, 5]) / self.cell_size)])

        print(pred_box_coordinate[..., 0])
        print(tf.nn.sigmoid(pred_box_coordinate[..., 0]))

        box = tf.stack([(tf.nn.sigmoid(pred_box_coordinate[..., 0]) + self.offset) / self.GRID_H,                                   # x 좌표만
                        (tf.nn.sigmoid(pred_box_coordinate[..., 1]) + tf.transpose(self.offset, [0, 2, 1, 3]) / self.GRID_H),       # y 좌표만
                        tf.sqrt(tf.exp(pred_box_coordinate[..., 2]) * np.reshape(self.ANCHOR[:5], [1, 1, 1, 5]) / self.GRID_H),     # w 너비만
                        tf.sqrt(tf.exp(pred_box_coordinate[..., 3]) * np.reshape(self.ANCHOR[5:], [1, 1, 1, 5]) / self.GRID_H)])    # h 높이만

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


    def build_modle(self):
        self.INPUT_X = tf.placeholder(dtype=tf.float32, shape=[None, self.IMG_H, self.IMG_W, self.IMG_C])
        self.INPUT_Y = tf.placeholder(dtype=tf.float32, shape=[None, self.GRID_H, self.GRID_W, self.N_ANCHORS, 5+self.N_CLASSES])

        self.logits = self.model(self.INPUT_X)
        self.loss = self.loss_layer(self.logits, self.INPUT_Y)

        self.optimizer = tf.train.AdamOptimizer(self.L_RATE).minimize(self.loss)
        

    def train_model(self):
        data_set_path = read_yolo_data_path(self.TRAIN_DATA_IMG_PATH, self.TRAIN_DATA_ANN_PATH)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.N_EPOCH):
                random.shuffle(data_set_path)           # 매 epoch마다 데이터셋 shuffling

                for i in range(int(len(data_set_path) / 2)):
                    batch_img_path, batch_ann_path = yolo_next_batch(data_set_path, i, self.N_BATCH)
                    batch_imgs = yolo_read_image(batch_img_path, self.N_BATCH, self.IMG_H, self.IMG_W)
                    
                    batch_anns = yolo_read_annot(batch_ann_path, self.N_BATCH, self.GRID_H, self.N_ANCHORS, self.N_CLASSES, self.IMG_H)
                    

                    _ ,loss_val = sess.run([self.optimizer, self.loss], feed_dict={self.INPUT_X:batch_imgs, self.INPUT_Y:batch_anns})
                    # self.writer.add_summary(summary_str, counter)
                    # counter += 1

                print('EPOCH: {}\t'.format(epoch+1), 'LOSS: {:.8}\t'.format(loss_val))

                output_check = sess.run([self.logits], feed_dict={self.INPUT_X:batch_imgs[0]})
                output_path = batch_img_path[0]

                test(output_check, output_path)


model = YOLO_V2()
model.build_modle()
model.train_model()
