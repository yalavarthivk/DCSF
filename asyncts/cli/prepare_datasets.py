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
            'mini_physionet'
            'mimic3_mortality',
            'LSST_async',
            'LSST_0_1',
            'LSST_0_5',
            'LSST_0_9',
            'PS_async',
            'PS_0_1',
            'PS_0_5',
            'PS_0_9',
            'activity'
        ],
        type=str
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        Normalizer(dataset)


if __name__ == '__main__':
    main()
