# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import os
import math

from hyperparameter import Hyperparameter as hp


def label_to_vec(label_str):
    """
    convert label string to index.
    :param label_str: str, disease label string, which can contain multiple diseases separated by '|'
    :return vec: np.array, k-hot vector indicating disease label
    """
    vec = np.zeros(len(hp.LABEL_TO_INDEX), dtype='int32')
    for label in label_str.split('|'):
        idx = hp.LABEL_TO_INDEX[label]
        vec[idx] = 1
    return vec


def load_label_dict(csv_path):
    """
    load `Data_Entry_2017.csv` and create dict of {image path: label string}
    :param csv_path: str, path to `Data_Entry_2017.csv`
    :return label_dict: dict, {image path: label string}
    """
    df = pd.read_csv(csv_path)
    label_dict = {row['Image Index']: row['Finding Labels'] for _, row in df.iterrows()}
    return label_dict


def make_example(image_path, label_str, rgb=True):
    """
    make TFRecord example from image_path(str) and label_str(str).
    :param image_path: str, path to image file
    :param label_str: str, disease label string, which can contain multiple diseases separated by '|'
    :param rgb: bool, whether to load image in RGB mode. channel=3 if True and otherwise channel=1
    :return example: instance of tf.train.Example
    """
    with Image.open(image_path) as image:
        if rgb:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        image = image.tobytes()
        label = label_to_vec(label_str)
        feature = {
            'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def save_as_tfrecords(image_path_list, save_path_format, label_dict, num_shard=1, rgb=True):
    """
    save data in TFRecord format.
    :param image_path_list: list, list of image_path (these paths don't have prefix, so we add hp.IMAGE_DIR)
    :param save_path_format: str, save file path format
    :param label_dict: dict, {image path: label string}
    :param num_shard: int, numer of output file
    :param rgb: bool, whether to load image in RGB mode. channel=3 if True and otherwise channel=1
    """
    data_size = len(image_path_list)
    shard_size = math.ceil(data_size / num_shard)
    for i in range(num_shard):
        save_path = str(save_path_format).format(i)
        writer = tf.python_io.TFRecordWriter(save_path)
        image_paths = image_path_list[i*shard_size:(i+1)*shard_size]
        print('[Info] saving {:,} files to {} ...'.format(len(image_paths), save_path))
        for image_path in image_paths:
            label_str = label_dict[str(image_path)]
            abs_image_path = hp.IMAGE_DIR / image_path
            example = make_example(abs_image_path, label_str, rgb)
            writer.write(example.SerializeToString())


def text_to_list(txt_file):
    """
    open image_path text file and convert it to list.
    :param txt_file: str, path to text file
    :return image_path_list: list, list of image_path
    """
    with open(txt_file) as f:
        image_path_list = [line.rstrip('\n') for line in f]
    return image_path_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default=hp.TFRECORDS_DIR, help='path to output dataset')
    parser.add_argument('-s', '--num_shard', type=int, default=1, help='numer of output file')
    parser.add_argument('-l', '--limit', type=int, default=None, help='limit the dataset size. for debug use')
    parser.add_argument('--gray', action='store_true', help='keep 1 channel image, not expand to 3 channel RGB image')
    args = parser.parse_args()

    # Image open mode
    rgb = False if args.gray else True

    # define output directory
    save_dir = Path(args.output)
    os.makedirs(str(save_dir), exist_ok=True)

    # load dict of {image_path: label}
    label_dict = load_label_dict(hp.data_entry_path)

    # load image path list
    train_image_path_list = text_to_list(str(hp.train_val_list_path))
    train_image_path_list, valid_image_path_list = train_test_split(
        train_image_path_list, train_size=int(len(train_image_path_list)*hp.train_ratio), random_state=1234)
    test_image_path_list = text_to_list(str(hp.test_list_path))

    # limit dataset size (for debug use)
    if args.limit is not None:
        train_image_path_list = train_image_path_list[:args.limit]
        valid_image_path_list = valid_image_path_list[:args.limit//args.num_shard]
        test_image_path_list = test_image_path_list[:args.limit//args.num_shard]

    # save as tfrecord
    train_save_path_format = save_dir / 'train_{:02d}.tfrecord'
    valid_save_path_format = save_dir / 'valid_{:02d}.tfrecord'
    test_save_path_format = save_dir / 'test_{:02d}.tfrecord'
    save_as_tfrecords(train_image_path_list, train_save_path_format, label_dict, args.num_shard, rgb)
    save_as_tfrecords(valid_image_path_list, valid_save_path_format, label_dict, 1, rgb)
    save_as_tfrecords(test_image_path_list, test_save_path_format, label_dict, 1, rgb)

    # get image size
    with Image.open(hp.IMAGE_DIR / train_image_path_list[0]) as image:
        image = np.array(image)
        height = image.shape[0]
        width = image.shape[1]
    channel = 3 if rgb else 1

    # save size information
    size_info = {
        'data_size': {
            'train': len(train_image_path_list),
            'valid': len(valid_image_path_list),
            'test': len(test_image_path_list),
        },
        'image_size': {
            'height': height,
            'width': width,
            'channel': channel,
        }
    }
    joblib.dump(size_info, save_dir / 'info.pkl')


if __name__ == '__main__':
    main()
