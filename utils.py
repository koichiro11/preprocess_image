# coding: utf-8
"""
hyperparameter
"""
import tensorflow as tf
import numpy as np


def data_augmentation(image, resize_h, resize_w, input_h, input_w, channel, is_training):
    """
    data resize & augmentation function for AG-CNN training phase.
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
        # random size cropping
        image = tf.image.resize_images(image, [resize_h, resize_w])
        image = tf.random_crop(image, [input_h, input_w, channel])

        # whiting(standardization): mean subtraction
        image = tf.image.per_image_standardization(image)

        # horizontal Flip
        image = tf.image.random_flip_left_right(image)


    else:
        image = tf.image.resize_images(image, [resize_h, resize_w])
        offset_h = (resize_h - input_h) // 2
        offset_w = (resize_w - input_w) // 2
        image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, input_h, input_w)
        image = tf.image.per_image_standardization(image)

    return image


def random_erasing(image, p=0.5, s_l=0.02, s_h=0.4, r1=0.3, r2=1. / 0.3):
    """
    :param image: tf.Tensor(dtype=tf.float32, shape=(h, w, c))
    :param p: probability to random erasing
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
        H = image.shape[0]
        W = image.shape[1]
        S = H * W

        while True:
            # maskする割合の決定(2~40%をmaskする)
            S_e = S * np.random.uniform(low=s_l, high=s_h)

            # アスペクト比の決定
            r_e = np.random.uniform(low=r1, high=r2)
            H_e = np.sqrt(S_e * r_e)
            W_e = np.sqrt(S_e / r_e)

            # 位置の決定
            x_e = np.random.randint(0, W)
            y_e = np.random.randint(0, H)

            # update
            if x_e + W_e <= W and y_e + H_e <= H:
                mask = tf.pad(tf.ones([int(H_e), int(W_e)]), [[y_e, H - H_e - y_e], [x_e, W - W_e - x_e]])


def random_erasing(img, p=0.5, s_l=0.02, s_h=0.4, r1=0.3, r2=1. / 0.3):
    """
    numpy
    :param img:
    :param p:
    :param s_l:
    :param s_h:
    :param r1:
    :param r2:
    :return:
    """
    # random eraseをするかどうか
    p1 = np.random.uniform(0, 1)
    if p1 < p:
        return img
    else:
        H = img.shape[0]
        W = img.shape[1]
        S = H * W
        while True:
            # maskする割合の決定(2~40%をmaskする)
            S_e = S * np.random.uniform(low=s_l, high=s_h)

            # アスペクト比の決定
            r_e = np.random.uniform(low=r1, high=r2)
            H_e = np.sqrt(S_e * r_e)
            W_e = np.sqrt(S_e / r_e)

            # 位置の決定
            x_e = np.random.randint(0, W)
            y_e = np.random.randint(0, H)

            if x_e + W_e <= W and y_e + H_e <= H:
                img_erased = np.copy(img)
                img_erased[y_e:int(y_e + H_e + 1), x_e:int(x_e + W_e + 1), :] = np.random.uniform(0, 1)
                return img_erased