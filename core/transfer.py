import tensorflow as tf
import numpy as np
import os

from core.graph import init_graph
from adain.image import load_image, prepare_image, save_image


def style_transfer(
        content_img=None,
        content_size=512,
        style_img=None,
        style_size=512,
        crop=None,
        alpha=1.0,
        content_dir='content',
        style_dir='style',
        output_dir='output',
        vgg_weights='models/vgg19_weights_normalized.h5',
        decoder_weights='models/decoder_weights.h5'):

    decoder_in_h5 = decoder_weights.endswith('.h5')

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    data_format = 'channels_last'

    if not os.path.exists(content_dir):
        os.mkdir(content_dir)
    if not os.path.exists(style_dir):
        os.mkdir(style_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image, content, style, target, encoder, decoder = init_graph(vgg_weights,
                                                                 decoder_weights if decoder_in_h5 else None, alpha,
                                                                 data_format=data_format)

    with tf.Session() as sess:
        if decoder_in_h5:
            sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, decoder_weights)

        content_name = content_img.filename
        style_name = style_img.filename
        content_path = os.path.join(content_dir, content_name)
        style_path = os.path.join(style_dir, style_name)
        with open(content_path, "wb") as f:
            f.write(content_img.read())
        with open(style_path, "wb") as f:
            f.write(style_img.read())
        content_image = load_image(content_path, content_size, crop)
        style_image = load_image(style_path, style_size, crop)

        style_image = prepare_image(style_image)
        content_image = prepare_image(content_image)
        style_feature = sess.run(encoder, feed_dict={
            image: style_image[np.newaxis, :]
        })
        content_feature = sess.run(encoder, feed_dict={
            image: content_image[np.newaxis, :]
        })
        target_feature = sess.run(target, feed_dict={
            content: content_feature,
            style: style_feature
        })
        output = sess.run(decoder, feed_dict={
            content: content_feature,
            target: target_feature
        })

        name = f"{content_name.split('.')[0]}_stylized_{style_name.split('.')[0]}.jpg"
        filename = os.path.join(output_dir, name)
        save_image(filename, output[0], data_format=data_format)
        return name.split('.')[0], filename
