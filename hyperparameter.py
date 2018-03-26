# coding: utf-8
"""
hyperparameter
"""
from pathlib import Path
import os


class HyperParameter(object):
    """
    hyper-parameter cloass
    """
    DATA_DIR = Path('/datadrive/CIFAR-10/')
    os.makedirs(str(DATA_DIR), exist_ok=True)

    train_X_path = DATA_DIR / 'train_X.pkl'
    train_y_path = DATA_DIR / 'train_y.pkl'
    test_X_path = DATA_DIR / 'test_X.pkl'
    test_y_path = DATA_DIR / 'test_y.pkl'

    train_ratio = 0.875  # train:valid(:test) = 7:1(:2)
    random_state = 1234
