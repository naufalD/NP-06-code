import tensorflow as tf
from read_batch import *
from Googlenet import *
import tools
import math
import os

batch_size = 8
image_size = 256
number_classes = 5
learning_rate = 0.001
MAX_STEP = 10000
train_log_dir = r"C:\Projects\Programming\Retina\logs\log2\train"
val_log_dir = r"C:\Projects\Programming\Retina\logs\log2\val"
dataset_directory = r"C:\Projects\Programming\Retina\train_fragments"

def train():
    image, labels= tf_read_batch(dataset_directory, batch_size)

    network = GNet(image, number_classes, True)
    loss = tools.loss(network, labels)
    accuracy = tools.accuracy(network, labels)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tools.optimize(loss, learning_rate, global_step)
    print("Network initialization check!")

    image_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3])
    label_placeholder = tf.placeholder(dtype=tf.int32, shape=[batch_size, number_classes])

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    print("Session initialization check!")

    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    print("Writer initialization check!")
    print("Starting training!")

    try:
        for step in np.arange(MAX_STEP):
            if coordinator.should_stop():
                break
            train_images, train_labels = sess.run([image, labels])
            _, train_loss, train_accuracy = sess.run([optimizer, loss, accuracy],
                                                     feed_dict={image_placeholder: train_images, label_placeholder: train_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print("Step: %d, loss: %.4f, accuracy: %.4f%%" % (step, train_loss, train_accuracy))
                summary_str = sess.run(summary_op)
                train_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([image, labels])
                val_loss, val_accuracy = sess.run([loss, accuracy],
                                                  feed_dict={image_placeholder: val_images, label_placeholder: val_labels})
                print("** Step: %d, loss: %.4f, accuracy: %.4f%%" % (step, val_loss, val_accuracy))
                summary_str = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, save_path=checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limited reached')
    finally:
        coordinator.request_stop()

    coordinator.join(threads)
    sess.close()


def evaluate():
    with tf.Graph().as_default():
        log_dir = r'C:\Projects\Programming\Retina\logs\trained_in_desktop\log2\train'
        test_dir = r'C:\Projects\Programming\Retina\Dataset\test_fragments'
        n_test = 53576

        test_image_batch, test_filename_batch = tf_read_test(test_dir, batch_size)

        image_placeholder = tf.placeholder(dtype=tf.float16, shape=[batch_size, image_size, image_size, 3])
        filenames_placeholder = tf.placeholder(dtype=tf.quint8, shape=[batch_size])

        logits = GNet(image_placeholder, number_classes, False)
        results = tools.results(logits)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            print('Reading checkpoint...')
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, os.path.join(log_dir,'model.ckpt-'+str(global_step)))
                print('Load success, global step: %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('\nEvaluating...')
                num_step = int(math.ceil(n_test / batch_size))
                print (num_step)
                num_example = num_step * batch_size
                step = 0
                with open(r'C:\Projects\Programming\Retina\Network3_results.csv', 'w', newline='') as csvfile:
                    fieldnames = ['image', 'level']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    while step < num_step and not coord.should_stop():
                        test_images, filenames = sess.run([test_image_batch, test_filename_batch])
                        result= sess.run(results, feed_dict={image_placeholder:test_images})
                        for x in range(batch_size):
                            new_filename = os.path.splitext(filenames[x].decode())[0]
                            new_result = result[x]
                            print(new_filename, new_result)
                            #print(new_result)
                            writer.writerow({'image': new_filename, 'level': new_result})
                        step += 1
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
    print("Done!")
evaluate()