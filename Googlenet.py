import tensorflow as tf
import tools

def GNet(x, n_class, is_pretrain):
    with tf.name_scope("GNet"):
        x = tools.conv('Convolution_1a', x, 64, [7,7], [1, 1, 1, 1])
        with tf.name_scope("Max_1"):
            x = tools.pool("Max_1", x, [1,3,3,1], [1,2,2,1], True)
        x = tools.conv('Convolution_2a', x, 64, [1,1], [1, 1, 1, 1])
        x = tools.conv('Convolution_2b', x, 192, [3,3], [1, 1, 1, 1])
        with tf.name_scope("Max_2"):
            x = tools.pool("Max_2", x, [1,3,3,1], [1,2,2,1], True)
        x = inception('Inception_3a', x, 64, 96, 128, 16, 32, 32)
        x = inception('Inception_3b', x, 128, 128, 192, 32, 96, 64)
        with tf.name_scope("Max_3"):
            x = tools.pool("Max_3", x, [1,3,3,1], [1,2,2,1], True)
        x = inception('Inception_4a', x, 192, 96, 208, 16, 48, 64)
        x = inception('Inception_4b', x, 160, 112, 224, 24, 64, 64)
        x = inception('Inception_4c', x, 128, 128, 256, 24, 64, 64)
        x = inception('Inception_4d', x, 112, 144, 288, 32, 64, 64)
        x = inception('Inception_4e', x, 256, 160, 320, 32, 128, 128)
        with tf.name_scope("Max_4"):
            x = tools.pool("Max_4", x, [1,3,3,1], [1,2,2,1], True)
        x = inception('Inception_5a', x, 64, 96, 128, 16, 32, 32)
        x = inception('Inception_5b', x, 128, 128, 192, 32, 96, 64)
        with tf.name_scope("Max_5"):
            x = tools.pool("Max_5", x, [1,3,3,1], [1,2,2,1], True)
        x = inception('Inception_6a', x, 64, 96, 128, 16, 32, 32)
        x = inception('Inception_6b', x, 128, 128, 192, 32, 96, 64)
        with tf.name_scope("Avg"):
            x = tools.pool("Avg", x, [1,8,8,1], [1,1,1,1], False)
        x = tools.FC_layer('fc6', x, out_nodes=128)
        with tf.name_scope('batch_norma1'):
            x = tools.batch_norm(x)
        x = tools.FC_layer('fc8', x, out_nodes=n_class)

        return x

def inception(layer_name, x, convo1, convo3_reduce, convo3, convo5_reduce, convo5, pool_reduce):
    with tf.variable_scope(layer_name):
        conv1 = tools.conv("Conv1", x, convo1, [1,1], [1,1,1,1], True)

        conv3 = tools.conv("Conv3_reduce", x, convo3_reduce, [1,1], [1,1,1,1], True)
        conv3 = tools.conv("Conv3", conv3, convo3, [3,3], [1,1,1,1], True)

        conv5 = tools.conv("Conv5_reduce", x, convo5_reduce, [1,1], [1,1,1,1], True)
        conv5 = tools.conv("Conv5", conv5, convo5, [5,5], [1,1,1,1], True)

        pool = tools.pool("Max_pool", x, [1,3,3,1], [1,1,1,1])
        pool = tools.conv("Pool_reduce", pool, pool_reduce, [1,1], [1,1,1,1], True)

        x = tf.concat([conv1, conv3, conv5, pool], 3, "Concatenate")

        return x

def nope():
    with tf.name_scope("Max_5"):
            x = tools.pool("Max_5", x, [1,3,3,1], [1,2,2,1], True)
    x = inception('Inception_6a', x, 96, 48, 104, 8, 24, 32)
    x = inception('Inception_6b', x, 80, 56, 112, 12, 32, 32)
    x = inception('Inception_6c', x, 64, 64, 128, 12, 32, 32)
    x = inception('Inception_6d', x, 56, 72, 144, 16, 32, 32)
    x = inception('Inception_6e', x, 128, 80, 160, 16, 64, 164)
    with tf.name_scope("Max_6"):
        x = tools.pool("Max_6", x, [1,3,3,1], [1,2,2,1], True)
    x = inception('Inception_7a', x, 256, 160, 320, 32, 128, 128)
    x = inception('Inception_7b', x, 384, 192, 384, 48, 128, 128)