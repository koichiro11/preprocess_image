# coding: utf-8
"""
create tf record
"""
import argparse
from data_loader import DataLoaderCIFAR10 as DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--num_shard', type=int, default=1, help='numer of output file')
    parser.add_argument('-l', '--limit', type=int, default=None, help='limit the dataset size. for debug use')

    args = parser.parse_args()

    data_loader = DataLoader()
    data_loader.main(num_shard=args.num_shard, limit=args.limit)
