"""Tensorflow datasets of medical time series."""
import medical_ts_datasets.checksums
import medical_ts_datasets.physionet_2012
import medical_ts_datasets.physionet_2019
import medical_ts_datasets.mimic_3_phenotyping
import medical_ts_datasets.mimic_3_mortality
import medical_ts_datasets.TSDB
import medical_ts_datasets.FaceDetection_0_5
import medical_ts_datasets.LSST_0_1
import medical_ts_datasets.PhonemeSpectra
import medical_ts_datasets.LSST_0_3
import medical_ts_datasets.LSST_0_5
import medical_ts_datasets.PS_0_1
import medical_ts_datasets.PS_0_3
import medical_ts_datasets.PS_0_5
import medical_ts_datasets.PS_0_7
import medical_ts_datasets.PS_0_9
import medical_ts_datasets.PS
import medical_ts_datasets.LSST_0_7
import medical_ts_datasets.LSST_0_9
import medical_ts_datasets.LSST
import medical_ts_datasets.Synth
import medical_ts_datasets.Synth_0_5
import medical_ts_datasets.Synth_0_1
import medical_ts_datasets.Synthnew_0_1
import medical_ts_datasets.Synth_0_9
import medical_ts_datasets.PenDigits_0_5
import medical_ts_datasets.activity
import medical_ts_datasets.LSST_async
import medical_ts_datasets.PS_async
import medical_ts_datasets.Synth_noci
import medical_ts_datasets.atari
import medical_ts_datasets.FD
import medical_ts_datasets.PD
import medical_ts_datasets.FD_async

builders = [
    'FD_async',
    'physionet2012',
    'physionet2019',
    'mimic3_mortality',
    'mimic3_phenotyping',
    'TSDB',
    'FaceDetection_0_5',
    'LSST_0_1',
    'PhonemeSpectra',
    'LSST_0_3',
    'LSST_0_5',
    'LSST_0_7',
    'LSST_0_9',
    'PS_0_1',
    'PS_0_3',
    'PS_0_5',
    'PS_0_7',
    'PS_0_9',
    'PS',
    'Synth',
    'PenDigits_0_5',
    'LSST',
    'activity',
    'LSST_async',
    'PS_async',
    'Synth_noci',
    'Synth_0_5',
    'Synth_0_1',
    'Synth_0_9',
    'atari',
    'Synthnew_0_1',
    'FD',
    'PD'
]

__version__ = '0.1.0'
