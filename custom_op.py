import tensorflow as tf

"""
Convolution layer
"""
def conv2d(inputs, depth, ksize, strides=[1, 1, 1, 1], padding='SAME', use_bias=True, initializer='xavier', name='conv2d'):
    if initializer == 'xavier':
        var_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    elif initializer == 'random':
        var_init = tf.random_normal_initializer(stddev=0.02)
    
    with tf.variable_scope(name):
        conv_w = tf.get_variable('conv_weight',
                                dtype=tf.float32,
                                shape=ksize+[inputs.get_shape().as_list()[-1], depth],
                                initializer=var_init)
                                
        if use_bias:
            conv_b = tf.get_variable('conv_bias',
                                     dtype=tf.float32,
                                     shape=[depth],
                                     initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d(inputs, conv_w, strides, padding), conv_b)
        else:
            return tf.nn.conv2d(inputs, conv_w, strides, padding)


def conv2d_t(inputs, out_shape, ksize, strides=[1, 2, 2, 1], use_bias=True, initializer='xavier', name='conv2d_transpose'):
    out_shape[0] = tf.shape(inputs)[0]
    ts_out_shape = tf.stack(out_shape)

    if initializer == 'xavier':
        var_init = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
    elif initializer == 'random':
        var_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope(name):
        convT_w = tf.get_variable('convT_weight',
                                  dtype=tf.float32, 
                                  shape=ksize+[out_shape[-1], inputs.get_shape()[-1]],
                                  initializer=var_init)
        if use_bias:
            convT_b = tf.get_variable('convT_bias', 
                                      dtype=tf.float32,
                                      shape=[out_shape[-1]],
                                      initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d_transpose(inputs, convT_w, ts_out_shape, strides), convT_b)
        else:
            return tf.nn.conv2d_transpose(inputs, convT_w, ts_out_shape, strides)


def atrous_conv2d(inputs, depth, ksize, rate, use_bias=True, name='atorus_conv'):
    with tf.variable_scope(name):
        atrous_w = tf.get_variable('atrous_weight',
                                    dtype=tf.float32,
                                    shape=ksize+[inputs.get_shape().as_list()[-1], depth], 
                                    initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        if use_bias:
            atrous_b = tf.get_variable('atrous_bias',
                                        dtype=tf.float32,
                                        shape=[depth],
                                        initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.atrous_conv2d(inputs, atrous_w, rate, 'SAME'), atrous_b)
        else:
            return tf.nn.atrous_conv2d(inputs, atrous_w, rate, 'SAME')


"""
Pooling layer
"""
def max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool'):
    return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding, name=name)

def avg_pool(inputs, ksize, strides, padding='SAME', name='avg_pool'):
    return tf.nn.avg_pool(inputs, ksize=ksize, strides=strides, padding=padding, name=name)


def fully_connect(inputs, out_dim, name='fc'):
    with tf.variable_scope(name):
        fc_w = tf.get_variable('fc_weight',
                                dtype=tf.float32,
                                shape=[inputs.get_shape().as_list()[-1], out_dim], 
                                initializer=tf.random_normal_initializer(stddev=0.02))
        
        fc_b = tf.get_variable('fc_bias',
                                dtype=tf.float32,
                                shape=[out_dim], 
                                initializer=tf.constant_initializer(0.0))

        return tf.matmul(inputs, fc_w) + fc_b


def bn(inputs, is_training):
    return tf.contrib.layers.batch_norm(inputs,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training)


"""
Activation Function
"""
def lrelu(inputs):
    return tf.nn.leaky_relu(inputs)

def relu(inputs):
    return tf.nn.relu(inputs)

def prelu(inputs, name=None):
    alpha = tf.get_variable(name + "/alpha", shape=[1], initializer=tf.constant_initializer(0), dtype=tf.float32)
    output = tf.nn.relu(inputs) + alpha*(inputs - abs(inputs))*0.5
    return output

def sigmoid(inputs):
    return tf.nn.sigmoid(inputs)


"""
Loss Function
"""
def softmax_with_logits(labels, pred, sentinel=None, dim=-1, name=None):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=pred)


"""
Dropout
"""
def spatial_dropout(inputs, keep_prob):
    out = tf.nn.dropout(inputs, keep_prob, noise_shape=[tf.shape(inputs)[0], 1, 1, tf.shape(inputs)[-1]])
    return out


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