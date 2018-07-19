import tensorflow as tf

def conv2d(inputs, out_dim, f_h, f_w, name, strides=[1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name):
        conv_w = tf.get_variable('conv_weight', dtype=tf.float32, shape=[f_h, f_w, inputs.get_shape()[-1], out_dim], 
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv_b = tf.get_variable('conv_bias', dtype=tf.float32, shape=[out_dim], 
                                initializer=tf.constant_initializer(0.0))
        
        return tf.nn.bias_add(tf.nn.conv2d(inputs, conv_w, strides=strides, padding=padding), conv_b)


def conv2d_t(inputs, out_shape, f_h, f_w, name, strides=[1, 2, 2, 1]):
    with tf.variable_scope(name):
        upconv_w = tf.get_variable('upconv_weight', dtype=tf.float32, shape=[f_h, f_w, out_shape[-1], inputs.get_shape()[-1]], 
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        upconv_b = tf.get_variable('upconv_bias', dtype=tf.float32, shape=[out_shape[-1]], 
                                initializer=tf.constant_initializer(0.0))
        
    return tf.nn.bias_add(tf.nn.conv2d_transpose(inputs, upconv_w, output_shape=out_shape, strides=strides, name=name), upconv_b)


def max_pool(inputs, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding, name=name)


def fully_connect(inputs, out_dim, name):
    with tf.variable_scope(name):
        fc_w = tf.get_variable('fc_weight', dtype=tf.float32, shape=[inputs.get_shape()[-1], out_dim], 
                                initializer=tf.random_normal_initializer(stddev=0.02))
        fc_b = tf.get_variable('fc_bias', dtype=tf.float32, shape=[out_dim], 
                                initializer=tf.constant_initializer(0.0))

        return tf.matmul(inputs, fc_w) + fc_b


def bn(inputs, is_training):
    return tf.contrib.layers.batch_norm(inputs, decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        is_training=is_training)


def lrelu(inputs):
    return tf.nn.leaky_relu(inputs)
