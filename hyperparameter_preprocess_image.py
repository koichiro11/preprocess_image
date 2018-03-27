# coding: utf-8
"""
hyper-parameter class
You should select class of use by inheritance
"""
from pathlib import Path
import os


class HyperParameterDefault(object):
    """
    hyper-parameter class
    """
    name = "dataset"
    DATA_DIR = Path('/datadrive') / name
    IMAGE_DIR = DATA_DIR / 'original'
    SAVE_DIR = DATA_DIR / 'record'

    train_ratio = 0.875  # train:valid(:test) = 7:1(:2)
    random_state = 1234

    train_X_path = None
    train_y_path = None
    test_X_path = None
    test_y_path = None

    train_val_list_path = None
    test_list_path = None
    data_entry_path = None
    bbox_list_path = None
    LABEL_TO_INDEX = None


class HyperParameterCIFAR10(HyperParameterDefault):
    """
    hyper-parameter class
    """
    name = 'cifar-10'
    output_dims = 10
    DATA_DIR = Path('/datadrive') / name
    IMAGE_DIR = DATA_DIR / 'original'
    SAVE_DIR = DATA_DIR / 'record'

    train_X_path = IMAGE_DIR / 'train_X.pkl'
    train_y_path = IMAGE_DIR / 'train_y.pkl'
    test_X_path = IMAGE_DIR / 'test_X.pkl'
    test_y_path = IMAGE_DIR / 'test_y.pkl'


class HyperParameterAG(HyperParameterDefault):
    """
    hyper-parameter for AG dataset
    """
    name = 'chest_X_ray'
    output_dims = 15
    DATA_DIR = Path('/datadrive') / name
    os.makedirs(str(DATA_DIR), exist_ok=True)
    IMAGE_DIR = DATA_DIR / 'original'
    os.makedirs(str(IMAGE_DIR), exist_ok=True)
    SAVE_DIR = DATA_DIR / 'record'
    os.makedirs(str(SAVE_DIR), exist_ok=True)


    train_val_list_path = IMAGE_DIR / 'train_val_list.txt'
    test_list_path = IMAGE_DIR / 'test_list.txt'
    data_entry_path = IMAGE_DIR / 'Data_Entry_2017.csv'
    bbox_list_path = IMAGE_DIR / 'BBox_List_2017.csv'

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

class HyperParameter(HyperParameterCIFAR10):
    """
    HyperParameter class to use
    """
    def __init__(self):
        print("use hyper-parameter for %s" % self.name)
        os.makedirs(str(self.DATA_DIR), exist_ok=True)
        os.makedirs(str(self.IMAGE_DIR), exist_ok=True)
        os.makedirs(str(self.SAVE_DIR), exist_ok=True)
