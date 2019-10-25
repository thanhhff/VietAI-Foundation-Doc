import numpy as np
import tensorflow as tf

import vgg16
import utils
import cv2

img = utils.load_image("./test_data/tiger.jpeg")

batch = img.reshape((1, 224, 224, 3))

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [1, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob, conv1_1 = sess.run([vgg.prob, vgg.conv1_1], feed_dict=feed_dict)

        conv1_1 = conv1_1[0,:,:,0]
        cv2.imshow('A feature map in conv1_1',conv1_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        utils.print_prob(prob[0], './synset.txt')
