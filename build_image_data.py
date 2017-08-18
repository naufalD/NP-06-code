import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import read_batch
import os
import csv
import re


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


directory = r"C:\Projects\Programming\Retina\Dataset\Reshaped_test_256"

csv_data = read_batch.read_csv('test_filenames_new.csv')

for x in range(27):
    x1 = 2000 * x
    x2 = 2000 + x1
    print(x, x1, x2)
    data = csv_data[x1:x2, :]
    tfrecords_filename = 'C:\\Projects\\Programming\\Retina\\test_fragment_2000_' + \
        str(x) + '_256.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for index, file_name in data:
        try:
            print(index, file_name)
            image = plt.imread(directory + '\\' + file_name)
            image_string = image.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'filename': _bytes_feature(file_name.encode(encoding='UTF-8')),
                'image_raw': _bytes_feature(image_string)}))

            writer.write(example.SerializeToString())
        except:
            break
    writer.close()
    print('Next!')
