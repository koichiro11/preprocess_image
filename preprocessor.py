# coding: utf-8
"""
preprocess class
"""
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import random_ops, math_ops
from hyperparameter_preprocess_image import HyperParameter as hp


class PreProcessorDefault(object):
    """
    preprocessor

    """
    def __init__(self,
                 dataset=hp.name,
                 output_dims=hp.output_dims):
        """
        :param dataset: dataset name
        :param output_dims: output_dims of label
        """
        self.dataset = dataset
        self.output_dims = output_dims

    @staticmethod
    def get_iterator(dataset, batch_size, num_epochs, buffer_size):
        """
        get data iterator.
        :param dataset: tf.data.Dataset API
        :param batch_size: int, batch size
        :param num_epochs: int, number of epochs
        :param buffer_size: int, buffer size for shuffling data
        :return iterator: Iterator instance
        """

        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def load_tfrecords_dataset(self, dataset_path_format, image_size, num_parallel_calls=None):
        """
        load TFRecord format dataset.
        :param dataset_path_format: str, path to dataset file format
        :param image_size: dict, contains original image size information, height, width, and channel
        :param num_parallel_calls: int or None, number of parallel processes to load dataset
        :return dataset: tf.data.Dataset API
        """
        filenames = tf.matching_files(dataset_path_format)
        num_shard = tf.cast(tf.shape(filenames)[0], tf.int64)
        dataset = tf.data.Dataset.list_files(filenames).shuffle(num_shard)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=num_shard)
        h = image_size['height']
        w = image_size['width']
        c = image_size['channel']
        dataset = dataset.map(lambda x: self.parse_example(x, h, w, c), num_parallel_calls=num_parallel_calls)
        return dataset

    def parse_example(self, example_proto, h, w, c):
        """
        parse TFRecord example
        :param example_proto: serialized tf.train.Example
        :param h: int, height of image
        :param w: int, width of image
        :param c: int, channel of image
        :return image: tf.Tensor(dtype=tf.float32, shape=(h, w, c))
        :return label: tf.Tensor(dtype=tf.int32, shape=(N, n_class))
        """
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
                'label': tf.FixedLenFeature(shape=[self.output_dims], dtype=tf.int64),
            })

        # cast
        image = tf.decode_raw(features['image'], tf.uint8)
        image = self.cast(image, h, w, c)

        label = tf.cast(features['label'], tf.int32)

        return image, label

    @staticmethod
    def cast(image, h, w, c):
        """
        :param image: tensor
        :return: tensor casted
        """
        image = tf.cast(image, tf.float32)
        image /= 255.
        image = tf.reshape(image, [h, w, c])
        return image


class PreProcessorWithAugmentation(PreProcessorDefault):
    """
    pre-process for image recognition
    if you would like to use pre-process for specific image dataset, please inheritance this class
    """
    def __init__(self,
                 dataset=hp.name,
                 output_dims=hp.output_dims):
        """
        :param dataset: dataset name
        :param output_dims: output_dims of label
        """
        super().__init__(dataset, output_dims)

    def get_iterator(self, dataset, batch_size, num_epochs, buffer_size, aug_kwargs, num_parallel_calls=None):
        """
        get data iterator.
        :param dataset: tf.data.Dataset API
        :param batch_size: int, batch size
        :param num_epochs: int, number of epochs
        :param buffer_size: int, buffer size for shuffling data
        :param aug_kwargs: dict, keyword arguments for aug_func
            ex:)
                aug_kwargs = {
                    'resize_h': 40,
                    'resize_w': 40,
                    'input_h': 32,
                    'input_w': 32,
                    'channel': 3,
                    'is_training': True,
                }
        :param num_parallel_calls: int or None, number of parallel processes to load dataset
        :return iterator: Iterator instance
        """

        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(lambda i, l: (self.data_augmentation(i, **aug_kwargs), l),
                              num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    def data_augmentation(self, image, resize_h, resize_w, input_h, input_w, channel=3, is_training=True):
        """
        data resize & augmentation function
        please override this function for specific data_augmentation
        :param image: tf.Tensor(dtype=tf.float32, shape=(h, w, c))
        :param resize_h: int, height after resizing original images
        :param resize_w: int, width after resizing original images
        :param input_h: int, input height
        :param input_w: int, input width
        :param channel: int, input channel
        :param is_training: boolean, is training step or not
        :return image: tf.Tensor(dtype=tf.float32, shape=(h, w, c))
        """

        if is_training:
            # resize
            image = tf.image.resize_images(image, [resize_h, resize_w])
            # random size cropping
            image = tf.random_crop(image, [input_h, input_w, channel])

            # horizontal Flip
            image = tf.image.random_flip_left_right(image)

            # random erasing
            image = self.random_erasing(image, input_h, input_w, channel)

            # random brightness & contrast
            image = tf.image.random_brightness(image, max_delta=63)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

            # whiting(standardization): mean subtraction
            image = tf.image.per_image_standardization(image)

        else:
            # resize
            image = tf.image.resize_images(image, [resize_h, resize_w])
            # center size cropping
            offset_h = (resize_h - input_h) // 2
            offset_w = (resize_w - input_w) // 2
            image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, input_h, input_w)

            # whiting(standardization): mean subtraction
            image = tf.image.per_image_standardization(image)

        return image

    @staticmethod
    def random_erasing(image, input_h, input_w, channel, p=0.5, s_l=0.02, s_h=0.4, r1=0.3, r2=1. / 0.3):
        """
        random erasing for tf.record
        this function is proposed by [Random Erasing Data Augmentation]
        ref: https://arxiv.org/abs/1708.04896

        :param image: tf.Tensor(dtype=tf.float32, shape=(h, w, c))
        :param p: probability to random erasing
        :param input_h: size of height
        :param input_w: size of width
        :param channel: image channel
        :param s_l: min ratio of S
        :param s_h: max ration of S
        :param r1: ratio of aspect 1
        :param r2: ratio of aspect 2
        :return: random erased image
        """
        # random eraseをするかどうか
        p1 = np.random.uniform(0, 1)
        if p1 < p:
            return image
        else:
            S = input_h * input_w

            while True:
                # maskする割合の決定(2~40%をmaskする)
                S_e = S * np.random.uniform(low=s_l, high=s_h)

                # アスペクト比の決定
                r_e = np.random.uniform(low=r1, high=r2)
                H_e = int(np.sqrt(S_e * r_e))
                W_e = int(np.sqrt(S_e / r_e))

                # 位置の決定
                x_e = np.random.randint(0, input_w)
                y_e = np.random.randint(0, input_h)

                # update
                if x_e + W_e <= input_w and y_e + H_e <= input_h:
                    noise = tf.random_uniform([input_h, input_w, channel])
                    _mask = tf.expand_dims(tf.pad(tf.ones([H_e, W_e]), [[y_e, input_h - y_e - H_e], [x_e, input_w - x_e - W_e]]), axis=-1)
                    _unmask = 1.0 - _mask
                    return _mask * noise + _unmask * image


    @staticmethod
    def random_erasing(image, input_h, input_w, channel, p=0.5, s_l=0.02, s_h=0.4, r1=0.3, r2=1. / 0.3):
        """
        random erasing for tf.record
        this function is proposed by [Random Erasing Data Augmentation]
        ref: https://arxiv.org/abs/1708.04896

        :param image: tf.Tensor(dtype=tf.float32, shape=(h, w, c))
        :param p: probability to random erasing
        :param input_h: size of height
        :param input_w: size of width
        :param channel: image channel
        :param s_l: min ratio of S
        :param s_h: max ration of S
        :param r1: ratio of aspect 1
        :param r2: ratio of aspect 2
        :return: random erased image
        """

        def func1():
            return image

        def func2():
            # 面積比の決定
            S_e_0 = tf.random_uniform([], minval=input_h * input_w * s_l, maxval=input_h * input_w * s_h)

            # アスペクト比の決定
            r_e_0 = tf.random_uniform([], minval=r1, maxval=r2)
            H_e_0 = tf.cast(tf.sqrt(S_e_0 * r_e_0), tf.int32)
            W_e_0 = tf.cast(tf.sqrt(S_e_0 / r_e_0), tf.int32)

            # 位置の決定
            x_e_0 = tf.cast(random_ops.random_uniform([], 0, maxval=input_w - tf.cast(W_e_0, tf.float32)), tf.int32)
            y_e_0 = tf.cast(random_ops.random_uniform([], 0, maxval=input_h - tf.cast(H_e_0, tf.float32)), tf.int32)

            def cond(x_e, W_e, y_e, H_e):
                return (x_e + W_e > input_w) | (y_e + H_e > input_h)
                # return x_e + W_e > input_w

            def body(x_e, W_e, y_e, H_e):
                # 面積比の決定
                S_e = tf.random_uniform([], minval=input_h * input_w * s_l, maxval=input_h * input_w * s_h)

                # アスペクト比の決定
                r_e = tf.random_uniform([], minval=r1, maxval=r2)
                H_e = tf.cast(tf.sqrt(S_e * r_e), tf.int32)
                W_e = tf.cast(tf.sqrt(S_e / r_e), tf.int32)

                # 位置の決定
                x_e = tf.cast(random_ops.random_uniform([], 0, maxval=input_w - tf.cast(W_e, tf.float32)), tf.int32)
                y_e = tf.cast(random_ops.random_uniform([], 0, maxval=input_h - tf.cast(H_e, tf.float32)), tf.int32)
                return x_e, W_e, y_e, H_e

            x_e_f, W_e_f, y_e_f, H_e_f = tf.while_loop(
                cond,
                body,
                [x_e_0, W_e_0, y_e_0, H_e_0]
            )

            noise = tf.random_uniform([input_h, input_w, channel])
            _mask = tf.expand_dims(
                tf.pad(tf.ones([H_e_f, W_e_f]), [[y_e_f, input_h - y_e_f - H_e_f], [x_e_f, input_w - x_e_f - W_e_f]]),
                axis=-1)
            _unmask = 1.0 - _mask
            return _mask * noise + _unmask * image
            # return _mask

        p1 = random_ops.random_uniform([], 0, 1.0)

        return tf.cond(
            p1 < p,
            func1,
            func2
        )
