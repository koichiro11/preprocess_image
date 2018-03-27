# coding: utf-8
"""
create tf record
"""

from data_loader import DataLoaderCIFAR10 as DataLoader

if __name__ == '__main__':

    data_loader = DataLoader()
    data_loader.main()