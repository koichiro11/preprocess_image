# coding: utf-8
"""
create tf record
"""

import numpy as np
import pickle
import joblib
import math
from PIL import Image

import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split


from hyperparameter import HyperParameter as hp


class TFRecordCreatorDefault(object):
    """
    create tf.record for training
    if you would like to use pre-process for specific image dataset, please inheritance this class
    """
    def __init__(self):
        print("TFRecord")

    def main(self,
             num_shard=1,
             limit=None):
        """
        tf record main function.
        when dataset is save to `single` file.
        :param num_shard: number of output file
        :param limit: limit the dataset size. for debug use
        """

        # load original dataset (get numpy array)
        train_X, train_y, test_X, test_y = self.load_original_data()

        # adjustment images through PIL


        # create valid dataset
        train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                              train_size=int(len(train_X.shape[0])*hp.train_ratio),
                                                              random_state=hp.random_state)

        # limit dataset size (for debug use)
        if limit is not None:
            train_X = train_X[:limit]
            train_y = train_y[:limit]
            valid_X = valid_X[:limit // num_shard]
            valid_y = valid_y[:limit // num_shard]
            test_X  = test_X[:limit // num_shard]
            test_y  = test_y[:limit // num_shard]

        # save as tf-record
        train_save_path_format = hp.DATA_DIR / 'train_{:02d}.tfrecord'
        valid_save_path_format = hp.DATA_DIR / 'valid_{:02d}.tfrecord'
        test_save_path_format = hp.DATA_DIR / 'test_{:02d}.tfrecord'
        self.save_as_tfrecords(train_X, train_y, train_save_path_format, num_shard)
        self.save_as_tfrecords(valid_X, valid_y, valid_save_path_format, 1)
        self.save_as_tfrecords(test_X, test_y, test_save_path_format, 1)


    @staticmethod
    def load_original_data():
        """
        load original dataset.
        this function is for `CIFAR-10`. If you would like to other dataset, please override this function
        :return: numpy array of dataset
        """

        if hp.train_X_path.exists():
            print("load data from pickle")
            with open(hp.train_X_path, 'rb') as f:
                train_X = pickle.load(f)
            with open(hp.train_y_path, 'rb') as f:
                train_y = pickle.load(f)
            with open(hp.test_X_path, 'rb') as f:
                test_X = pickle.load(f)
            with open(hp.test_y_path, 'rb') as f:
                test_y = pickle.load(f)

        else:
            print("load data from keras.library")
            (_train_X, _train_y), (_test_X, _test_y) = cifar10.load_data()
            train_X, test_X = _train_X.astype('float32'), _test_X.astype('float32')
            train_y, test_y = np.eye(10)[_train_y.astype('int32').flatten()], np.eye(10)[_test_y.astype('int32').flatten()]

            with open(hp.train_X_path, 'rb') as f:
                pickle.dump(train_X, f)
            with open(hp.train_y_path, 'rb') as f:
                pickle.dump(train_y, f)
            with open(hp.test_X_path, 'rb') as f:
                pickle.dump(test_X, f)
            with open(hp.test_y_path, 'rb') as f:
                pickle.dump(test_y, f)

        return train_X, train_y, test_X, test_y

    def save_as_tfrecords(self, images, labels, save_path_format, num_shard=1):
        """
        save data in TFRecord format.
        :param images: numpy.ndarray
        :param labels: numpy.ndarray, label
        :param save_path_format: str, save file path format
        :param num_shard: int, number of output file
        """
        data_size = len(images)
        shard_size = math.ceil(data_size / num_shard)
        for i in range(num_shard):
            save_path = str(save_path_format).format(i)
            writer = tf.python_io.TFRecordWriter(save_path)
            _images = images[i * shard_size:(i + 1) * shard_size]
            _labels = labels[i * shard_size:(i + 1) * shard_size]
            print('[Info] saving {:,} files to {} ...'.format(len(_images), save_path))
            for image, label in zip(_images, _labels):
                labels = labels.astype(np.float32)
                ex = self.make_example(image, label)
                writer.write(ex.SerializeToString())

            writer.close()

    @staticmethod
    def make_example(image, label):
        """
        make TFRecord example from image and label
        :param image: numpy array
        :param label: numpy array
        :return example: instance of tf.train.Example
        """
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example

    @staticmethod
    def convert_from_pil_into_numpy(pil_image, rgb=True):
        """
        convert PIL into numpy array
        :param pil_image: PIL image
        :param rgb: bool, whether to load image in RGB mode. channel=3 if True and otherwise channel=1
        :return: converted image, numpy array
        """
        # convert to PIL
        if rgb:
            pil_image = pil_image.convert('RGB')
        else:
            pil_image = pil_image.convert('L')

        return np.asarray(pil_image)


class TFRecordCreatorAG(TFRecordCreatorDefault):
    """
    TFRecordCreator
    When you load image one by one
    """
    def __init__(self,
                 label_dict,
                 image_dir=hp.IMAGR_DIR,
                 save_dir=hp.DATASET_DIR,
                 label_to_index=hp.LABEL_TO_INDEX,
                 train_val_list_path=hp.train_val_list_path,
                 test_list_path=hp.test_list_path,
                 data_entry_path=hp.data_entry_path,
                 bbox_list_path=hp.bbox_list_path,
                 train_ratio=0.875,
                 ):
        """
        :param label_dict: dict, {image path: label string}
        :param image_dir: image directory ex: '/datadrive/chest_X_ray/images'
        :param save_dir: save directory for TFRecord ex: '~/chest_X_ray/'
        :param label_to_index: dict for label to index
            ex:
            LABEL_TO_INDEX = {
                    'No Finding': 0,
                    'Atelectasis': 1,
                    'Cardiomegaly': 2,
                    'Effusion': 3,
                    'Infiltration': 4,
                    'Mass': 5,
                    'Nodule': 6,
                    'Pneumonia': 7,
                    'Pneumothorax': 8,
                    'Consolidation': 9,
                    'Edema': 10,
                    'Emphysema': 11,
                    'Fibrosis': 12,
                    'Pleural_Thickening': 13,
                    'Hernia': 14,
                }
        :param train_val_list_path: ex: '/datadrive/chest_X_ray/train_val_list.txt'
        :param train_val_list_path: ex: '/datadrive/chest_X_ray/test_list.txt'
        :param train_val_list_path: ex: '/datadrive/chest_X_ray/Data_Entry_2017.csv'
        :param train_val_list_path: ex: '/datadrive/chest_X_ray/BBox_List_2017.csv'
        :param train_ratio: train_ratio
        """
        super().__init__()
        self.label_dict = label_dict
        self.IMAGE_DIR = image_dir
        self.save_dir = save_dir
        self.LABEL_TO_INDEX = label_to_index
        self.train_val_list_path = train_val_list_path
        self.test_list_path = test_list_path
        self.data_entry_path = data_entry_path
        self.bbox_list_pathbbox_list_path = bbox_list_path
        self.train_ratio = train_ratio

    def main(self,
             num_shard=1,
             limit=None,
             rgb=True):
        """
        tf record main function.
        when dataset is save to `single` file.
        :param num_shard: number of output file
        :param limit: limit the dataset size. for debug use
        :param rgb: bool, whether to load image in RGB mode. channel=3 if True and otherwise channel=1
        """

        # load image path list
        train_image_path_list = self.text_to_list(str(self.train_val_list_path))
        train_image_path_list, valid_image_path_list = train_test_split(
            train_image_path_list, train_size=int(len(train_image_path_list) * self.train_ratio), random_state=1234)
        test_image_path_list = self.text_to_list(str(self.test_list_path))

        # limit dataset size (for debug use)
        if limit is not None:
            train_image_path_list = train_image_path_list[:limit]
            valid_image_path_list = valid_image_path_list[:limit // num_shard]
            test_image_path_list = test_image_path_list[:limit // num_shard]

        # save as tfrecord
        train_save_path_format = self.save_dir / 'train_{:02d}.tfrecord'
        valid_save_path_format = self.save_dir / 'valid_{:02d}.tfrecord'
        test_save_path_format = self.save_dir / 'test_{:02d}.tfrecord'
        self.save_as_tfrecords(train_image_path_list, train_save_path_format, self.label_dict, num_shard, rgb)
        self.save_as_tfrecords(valid_image_path_list, valid_save_path_format, self.label_dict, 1, rgb)
        self.save_as_tfrecords(test_image_path_list, test_save_path_format, self.label_dict, 1, rgb)

        # get image size
        with Image.open(self.IMAGE_DIR / train_image_path_list[0]) as image:
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
        joblib.dump(size_info, self.save_dir / 'info.pkl')

    def save_as_tfrecords(self, image_path_list, save_path_format, num_shard=1, rgb=True):
        """
        save data in TFRecord format.
        :param image_path_list: list, list of image_path (these paths don't have prefix, so we add config.IMAGE_DIR)
        :param save_path_format: str, save file path format
        :param num_shard: int, numer of output file
        :param rgb: bool, whether to load image in RGB mode. channel=3 if True and otherwise channel=1
        """
        data_size = len(image_path_list)
        shard_size = math.ceil(data_size / num_shard)
        for i in range(num_shard):
            save_path = str(save_path_format).format(i)
            writer = tf.python_io.TFRecordWriter(save_path)
            image_paths = image_path_list[i * shard_size:(i + 1) * shard_size]
            print('[Info] saving {:,} files to {} ...'.format(len(image_paths), save_path))
            for image_path in image_paths:
                # image
                abs_image_path = self.IMAGE_DIR / image_path
                with Image.open(abs_image_path) as image:
                    image = self.convert_from_pil_into_numpy(image, rgb=True)

                # label
                label_str = self.label_dict[str(image_path)]
                label = self.label_to_vec(label_str)

                example = self.make_example(image, label)
                writer.write(example.SerializeToString())

            writer.close()

    def label_to_vec(self, label_str):
        """
        convert label string to index.
        :param label_str: str, disease label string, which can contain multiple diseases separated by '|'
        :return vec: np.array, k-hot vector indicating disease label
        """
        vec = np.zeros(len(self.LABEL_TO_INDEX), dtype='int32')
        for label in label_str.split('|'):
            idx = self.LABEL_TO_INDEX[label]
            vec[idx] = 1
        return vec

    @staticmethod
    def text_to_list(txt_file):
        """
        open image_path text file and convert it to list.
        :param txt_file: str, path to text file
        :return image_path_list: list, list of image_path
        """
        with open(txt_file) as f:
            image_path_list = [line.rstrip('\n') for line in f]
        return image_path_list

