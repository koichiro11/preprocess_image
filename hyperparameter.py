# coding: utf-8
"""
hyperparameter
"""
from pathlib import Path
import os


class Hyperparameter(object):
    MAIN_DIR = Path(os.environ['HOME']) / 'CIFAR-10'
    DATASET_DIR = MAIN_DIR / 'dataset/'
    OUTPUT_DIR = MAIN_DIR / 'output/'

    DATA_DIR_PATH = '/datadrive/CIFAR-10/'
    DATA_DIR = Path(DATA_DIR_PATH)
    IMAGE_DIR = DATA_DIR / 'images'

    train_X_path = DATA_DIR / 'train_X.pkl'
    train_y_path = DATA_DIR / 'train_y.pkl'
    test_X_path = DATA_DIR / 'test_X.pkl'
    test_y_path = DATA_DIR / 'test_y.pkl'

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

    train_ratio = 0.875  # train:valid(:test) = 7:1(:2)
