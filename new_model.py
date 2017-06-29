# encoding: utf-8

# tensorflow
import tensorflow as tf
import math
from model_part import conv2d
from model_part import fullyConnectedLayer
TRAIN_FILE = "train.csv"
BATCH_SIZE = 8
from dataset import DataSet

def globalDepthMap(images, reuse=False, trainable=True):
    with tf.name_scope("Global_Depth"):
        pre_coarse1 = tf.layers.conv2d(inputs=images, filters=96, kernel_size=[11, 11], strides=4, padding='valid', activation=tf.nn.relu, name='pre_coarse1', reuse=tf.get_variable_scope().reuse)
        coarse1 = tf.layers.max_pooling2d(inputs=pre_coarse1, pool_size=[2, 2], strides=2, padding='valid', name='coarse1')
        pre_coarse2 = tf.layers.conv2d(inputs=coarse1, filters=256, kernel_size=[5, 5], strides=1, padding='same', activation=tf.nn.relu, name='pre_coarse2', reuse=tf.get_variable_scope().reuse)
        coarse2 = tf.layers.max_pooling2d(inputs=pre_coarse2, pool_size=[2, 2], strides=2, padding='valid', name='coarse2')
        coarse3 = tf.layers.conv2d(inputs=coarse2, filters=384, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu, name='coarse3', reuse=tf.get_variable_scope().reuse)
        coarse4 = tf.layers.conv2d(inputs=coarse3, filters=384, kernel_size=[3, 3], strides=1, padding='same', activation=tf.nn.relu, name='coarse4', reuse=tf.get_variable_scope().reuse)
        coarse5 = tf.layers.conv2d(inputs=coarse4, filters=256, kernel_size=[3, 3], strides=2, padding='valid', activation=tf.nn.relu, name='coarse5', reuse=tf.get_variable_scope().reuse)
        coarse5_flat = tf.reshape(coarse5, [-1, 6 * 8 * 256])
        coarse6 = tf.layers.dense(inputs=coarse5_flat , units=4096, activation=tf.nn.relu, reuse=tf.get_variable_scope().reuse)
        pre_coarse7 = tf.layers.dense(inputs=coarse6, units=4070, activation=tf.nn.relu, reuse=tf.get_variable_scope().reuse)
        coarse7 = tf.reshape( pre_coarse7, [-1, 55, 74, 1])
    
        #print("pre_coarse1", pre_coarse1._shape)
        #print("coarse1", coarse1._shape)
        #print("pre_coarse2", pre_coarse2._shape)
        #print("coarse2", coarse2._shape)
        #print("coarse3", coarse3._shape)
        #print("coarse4", coarse4._shape)
        #print("coarse5", coarse5._shape)
        #print("coarse6", coarse6._shape)
        #print("pre_coarse7",  pre_coarse7._shape)
        #print("coarse7", coarse7._shape)

    return coarse7

def localDepthMap(images, coarse7_output, keep_conv, reuse=False, trainable=True):
    with tf.name_scope("Local_Depth"):
        pre_fine1 = tf.layers.conv2d(inputs=images, filters=63, kernel_size=[9,9], strides=2, padding='valid', activation=tf.nn.relu, name='pre_fine1', reuse=tf.get_variable_scope().reuse)
        fine1 = tf.layers.max_pooling2d(inputs=pre_fine1, pool_size=[2, 2], strides=2, padding='same', name='fine1')
        fine1_dropout = tf.layers.dropout(inputs=fine1, rate=0.5, noise_shape=None, seed=None, training=trainable, name='fine1_dropout')
        fine2 = tf.concat(axis=3, values=[fine1_dropout, coarse7_output], name="fine2_concat")
        fine3 = tf.layers.conv2d(inputs=fine2, filters=64, kernel_size=[5,5], strides=1, padding='same', activation=tf.nn.relu, name='fine3', reuse=tf.get_variable_scope().reuse)
        fine3_dropout = tf.layers.dropout(inputs=fine3, rate=0.5, noise_shape=None, seed=None, training=trainable, name='fine3_dropout')
        fine4 = tf.layers.conv2d(inputs=fine3_dropout, filters=1, kernel_size=[5,5], strides=1, padding='same', activation=tf.nn.relu, name='fine4', reuse=tf.get_variable_scope().reuse)
        fine4_full = tf.layers.dense(inputs=fine4, units=4070, activation=tf.nn.relu, reuse=tf.get_variable_scope().reuse)
        fine4_res = tf.reshape(fine4_full, [-1, 55, 74, 1])

        #print("pre_fine1 ", pre_fine1._shape)
        #print("fine1 ", fine1._shape)
        #print("fine1_dropout ", fine1_dropout._shape)
        #print("fine2 ", fine2._shape)
        #print("fine3 ", fine3._shape)
        #print("fine3_dropout ", fine3_dropout._shape)
        #print("fine4 ", fine4._shape)

    return fine4

def loss(logits, depths, invalid_depths):
    logits_flat = tf.reshape(logits, [-1, 55*74])
    depths_flat = tf.reshape(depths, [-1, 55*74])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 55*74])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(sum_square_d / 55.0*74.0 - 0.5*sqare_sum_d / math.pow(55*74, 2))
    tf.add_to_collection('losses', cost)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

#if __name__ == '__main__':
    #dataset = DataSet(BATCH_SIZE)
    #images, depths, invalid_depths = dataset.create_trainingbatches_from_csv(TRAIN_FILE)
    #print(images._shape)
    #print("----------------------------")
    #out = globalDepthMap(images)
    #localDepthMap(images, out, 4)