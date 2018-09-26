import cv2
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import xml.etree.ElementTree as ET

from PIL import Image
# opencv로 이미지를 읽으면 RGB 순서가 아닌 BGR 순서임.


def read_data_path(x, y):
    inputs_path = [os.path.join(x, file) for file in os.listdir(x)]
    labels_path = [os.path.join(y, file) for file in os.listdir(y)]

    assert len(inputs_path) == len(labels_path)

    trainSet_path = list()

    for i in range(len(inputs_path)):
        trainSet_path.append([inputs_path[i], labels_path[i]])

    return trainSet_path

def next_batch(trainSet_path, batch_size, idx):
    batchs = trainSet_path[idx*batch_size : idx*batch_size+batch_size]
    batch_xPath, batch_yPath = [], []
    
    for i in range(len(batchs)):
        batch_xPath.append(batchs[i][0])
        batch_yPath.append(batchs[i][1])

    return batch_xPath, batch_yPath

def read_image(path, resize):
    batch_x = np.zeros(shape=[len(path), resize[0], resize[1], 3])
    
    for i in range(len(path)):
        # image = misc.imread(path[i])
        image = cv2.imread(path[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = misc.imresize(image, resize, interp='nearest')
        #image = image / 255.0 * 2.0 - 1.0
        batch_x[i] = image

    return batch_x

## segmentation model 에서 사용할 annotation
def read_annotation(path, resize):
    batch_y = np.zeros(shape=[len(path), resize[0], resize[1], 1])

    for i in range(len(path)):
        # grayscale 로 읽을 때에는 mode='L'
        label = cv2.imread(path[i], cv2.IMREAD_GRAYSCALE)
        label = misc.imresize(label, resize, interp='nearest')
        label = np.expand_dims(label, axis=3)
        #label = label / 255.0 * 2.0 - 1.0
        batch_y[i] = label
    
    return batch_y

## yolo model 에서 사용할 annotation
def read_xml(path, n_batch, image_size, grid_size, n_bbox, n_class, class_info):
    batch_y = np.zeros([n_batch, grid_size, grid_size, n_bbox, 5+n_class])

    for i in range(n_batch):
        file = path[i]
        tree = ET.parse(file)

        xml_imageSize = tree.find('size')
        xml_imageHeight = float(xml_imageSize.find('height').text)
        xml_imageWidth = float(xml_imageSize.find('width').text)

        h_ratio = float(image_size / xml_imageHeight)
        w_ratio = float(image_size / xml_imageWidth)

        ## xml 에 object 태그를 모두 찾음
        objects = tree.findall('object')

        for obj in objects:
            bbox = obj.find('bndbox')

            ## bbox의 top-left 좌표
            x1 = max(min((float(bbox.find('xmin').text)) * w_ratio, image_size), 0)
            y1 = max(min((float(bbox.find('ymin').text)) * h_ratio, image_size), 0)
            ## bbox의 bottom-right 좌표
            x2 = max(min((float(bbox.find('xmax').text)) * w_ratio, image_size), 0)
            y2 = max(min((float(bbox.find('ymax').text)) * h_ratio, image_size), 0)

            ## obejct의 class
            class_idx = class_info.index(obj.find('name').text.lower().strip())

            ## bbox의 중심점 좌표
            box_coor = [0.5 * (x1 + x2) / image_size, 0.5 * (y1 + y2) / image_size, \
                        np.sqrt((x2 - x1) / image_size), np.sqrt((y2 - y1) / image_size)]

            cx = 1.0 * box_coor[0] * grid_size
            cy = 1.0 * box_coor[1] * grid_size

            cx_idx = int(np.floor(cx))
            cy_idx = int(np.floor(cy))

            batch_y[i, cy_idx, cx_idx, :, 0] = 1                # confidence
            batch_y[i, cy_idx, cx_idx, :, 1:5] = box_coor       # coordinate
            batch_y[i, cy_idx, cx_idx, :, 5+class_idx] = 1      # class info

    return batch_y


def draw_plot_segmentation(path, image, pred, gt):
    batch = len(image)
    image_list = list()
    fig = plt.figure()

    for i in range(batch):
        image_list.append(image[i])
        image_list.append(pred[i])
        image_list.append(gt[i])

    for i in range(len(image_list)):
        fig.add_subplot(batch, 3, i+1)
        plt.imshow(image_list[i].astype(np.uint8))

    fig.savefig(path, bbox_inches='tight')
    plt.close()

def draw_plot_gan(pred, save_path):
    fig = plt.figure()

    for i in range(len(pred)):
        fig.add_subplot(1, len(pred), i+1)
        plt.imshow(pred[i], cmap='gray')
        plt.axis('off')
    
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    