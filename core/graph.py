import tensorflow as tf
from adain.norm import adain
from adain.weights import open_weights
from adain.nn import build_vgg, build_decoder


def init_graph(vgg_weights, decoder_weights, alpha, data_format):
    if data_format == 'channels_first':
        image = tf.placeholder(shape=(None, 3, None, None), dtype=tf.float32)
        content = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
        style = tf.placeholder(shape=(1, 512, None, None), dtype=tf.float32)
    else:
        image = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32)
        content = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)
        style = tf.placeholder(shape=(1, None, None, 512), dtype=tf.float32)

    target = adain(content, style, data_format=data_format)
    weighted_target = target * alpha + (1 - alpha) * content

    with open_weights(vgg_weights) as w:
        vgg = build_vgg(image, w, data_format=data_format)
        encoder = vgg['conv4_1']

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = build_decoder(weighted_target, w, trainable=False,
                                    data_format=data_format)
    else:
        decoder = build_decoder(weighted_target, None, trainable=False,
                                data_format=data_format)

    return image, content, style, target, encoder, decoder
