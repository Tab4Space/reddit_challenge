import tensorflow as tf

"""
Convolution layer
"""
def conv2d(inputs, depth, ksize, strides=[1, 1, 1, 1], padding='SAME', use_bias=True, name='conv2d'):
    with tf.variable_scope(name):
        conv_w = tf.get_variable('conv_weight',
                                dtype=tf.float32,
                                shape=ksize+[inputs.get_shape().as_list()[-1], depth],
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
                                
        if use_bias:
            conv_b = tf.get_variable('conv_bias',
                                     dtype=tf.float32,
                                     shape=[depth],
                                     initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d(inputs, conv_w, strides, padding), conv_b)
        else:
            return tf.nn.conv2d(inputs, conv_w, strides, padding)


def conv2d_t(inputs, out_shape, ksize, strides=[1, 2, 2, 1], use_bias=True, name='conv2d_transpose'):
    with tf.variable_scope(name):
        convT_w = tf.get_variable('convT_weight',
                                  dtype=tf.float32, 
                                  shape=ksize+[out_shape[-1], inputs.get_shape().as_list()[-1]], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False))
        if use_bias:
            convT_b = tf.get_variable('convT_bias', 
                                      dtype=tf.float32,
                                      shape=[out_shape[-1]],
                                      initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d_transpose(inputs, convT_w, out_shape, strides), convT_b)
        else:
            return tf.nn.conv2d_transpose(inputs, convT_w, out_shape, strides)


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
                                        is_training=is_training)


"""
Activation Function
"""
def lrelu(inputs):
    return tf.nn.leaky_relu(inputs)

def relu(inputs):
    return tf.nn.relu(inputs)


"""
Loss Function
"""
def softmax_with_logits(labels, pred, sentinel=None, dim=-1, name=None):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=pred)