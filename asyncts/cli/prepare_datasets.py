"""Prepare all datasets."""
import argparse

import medical_ts_datasets
from seft.normalization import Normalizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'datasets',
        nargs='+',
        choices=[
            'physionet2012',
            'mimic3_mortality',
        ],
        type=str
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        Normalizer(dataset)


if __name__ == '__main__':
    main()