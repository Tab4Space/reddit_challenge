import numpy as np
import os
import scipy.misc as misc
import matplotlib.pyplot as plt

def load_data_path(image_dir, annotation_dir, mode):
    img_path = os.path.join(image_dir, mode)
    ann_path = os.path.join(annotation_dir, mode)

    images_file_name = [os.path.join(img_path, i) for i in os.listdir(img_path)]
    annots_file_name = [os.path.join(ann_path, i) for i in os.listdir(ann_path)]

    data_set = []

    for i in range(len(images_file_name)):
        img_shape = misc.imread(images_file_name[i]).shape
        ann_shape = misc.imread(annots_file_name[i]).shape

        if len(img_shape) == 3 and len(ann_shape) == 2:
            data_set.append([images_file_name[i], annots_file_name[i]])

    return data_set

def next_batch(data_set_path, idx, batch_size):
    batchs = data_set_path[idx*batch_size : idx*batch_size+batch_size]
    batch_images = []
    batch_annots = []
    for i in range(batch_size):
        batch_images.append(batchs[i][0])
        batch_annots.append(batchs[i][1])

    return batch_images, batch_annots

def read_image(batch_img_path, batch_ann_path, batch_size, resize_h, resize_w):
    imgs, anns = [], []
    for i in range(batch_size):
        img = misc.imread(batch_img_path[i])
        ann = misc.imread(batch_ann_path[i])

        img_resize = misc.imresize(img, [resize_h, resize_w], interp='nearest').reshape([1, resize_h, resize_w, 3])
        ann_resize = misc.imresize(ann, [resize_h, resize_w], interp='nearest').reshape([1, resize_h, resize_w])
        ann_resize = np.expand_dims(ann_resize, axis=3)

        imgs.append(img_resize)
        anns.append(ann_resize)

    return np.concatenate(imgs, axis=0), np.concatenate(anns, axis=0)


def plotting(image, gt, pred, savepath):
    plt.subplot(3, 1, 1)
    plt.imshow(image[0])
    plt.subplot(3, 1, 2)
    plt.imshow(pred)
    plt.subplot(3, 1, 3)
    plt.imshow(gt[0])
    
    plt.savefig(savepath, bbox_inches='tight')

# data_set_path = load_data_path('../dataset/images/', '../dataset/annotations/', 'training')
# a, b = next_batch(data_set_path, 0, 2)
# c, d = read_image(a, b, 2)

# print(c.shape)
# print(d.shape)