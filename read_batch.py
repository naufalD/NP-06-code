import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, csv 

def read_batch(batch_size, labels):
    image_batch = np.array([])
    directory = r"C:\Projects\Programming\Retina\Reshaped_train"
    
    mask = np.random.choice(labels.shape[0], batch_size, replace=False)
    image_directories = labels[mask, 0]
    labels = labels[mask]

    for file_name in image_directories:
        if (image_batch.shape[0]):
            image_batch = np.concatenate((image_batch, [plt.imread(directory+'\\'+file_name+'.jpeg')]), axis=0)
        else:
            image_batch =  np.array([plt.imread(directory+'\\'+file_name+'.jpeg')])
    labels = labels[:,1]
    labels = tf.one_hot(labels, 4, dtype=tf.float16)
    #image_batch, labels, filenames = tf.train.batch([image_batch, labels, filenames], batch_size, )
    return image_batch, labels, image_directories

def read_csv(filename):
    labels = np.array([[1,1]])
    with open(filename, 'rt') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels = np.append(labels, [row], axis=0)
    labels = labels[2:]
    return labels

def tf_read_batch(directory, batch_size, mode = 'train'):
    directory= [os.path.join(directory, 'fragment_2000_%d_256.tfrecords' % ii) for ii in np.arange(0, 10)]
    filename_queue = tf.train.input_producer(directory)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)

    features = tf.parse_single_example(
        example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    labels = tf.cast(features['label'], tf.uint8)

    image_shape = tf.stack([256,256,3])

    image = tf.reshape(image, image_shape)

    image, labels = tf.train.shuffle_batch( [image, labels],
                                                 batch_size=batch_size,
                                                 capacity=2000,
                                                 num_threads=64,
                                                 min_after_dequeue=100)

    labels = tf.one_hot(labels, depth=5)
    labels = tf.cast(labels, dtype=tf.int32)
    image = tf.cast(image, dtype=tf.float16)

    return image, labels

def tf_read_test(directory, batch_size):
    directory= [os.path.join(directory, 'test_fragment_2000_%d_256.tfrecords' % ii) for ii in np.arange(0,27)]
    print(directory)
    filename_queue = tf.train.input_producer(directory, shuffle=False)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)  

    features = tf.parse_single_example(
        example,
        features={
            'filename': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )  

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    filename = tf.cast(features['filename'], tf.string)

    image_shape = tf.stack([256,256,3])

    image = tf.reshape(image, image_shape)
    image, filename = tf.train.batch( [image, filename],
                                                 batch_size=batch_size,
                                                 capacity=60000,
                                                 num_threads=1,)
    image = tf.cast(image, dtype=tf.float16)
    return image, filename

