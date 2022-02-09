import tensorflow_datasets as tfds
import tensorflow as tf
import medical_ts_datasets
import pdb
import numpy as np
from collections.abc import Sequence
import time
autotune = tf.data.experimental.AUTOTUNE
from tensorflow import keras
from tensorflow.python.framework.smart_cond import smart_cond

get_output_shapes = tf.compat.v1.data.get_output_shapes
get_output_types = tf.compat.v1.data.get_output_types
make_one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator
from .cnn_model import cnn_simple_model_temp
from tensorflow.keras.layers import Dense
# from keras_transformer.attention import MultiHeadSelfAttention
from .preproc import data_processing
from .set_utils import (
    build_dense_dropout_model, PaddedToSegments, SegmentAggregation,
    cumulative_softmax_weighting, cumulative_segment_mean)
from .utils import segment_softmax
from itertools import chain
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os 
import sys
import tensorflow_probability as tfp


class RESNET_forward(tf.keras.Model):
	dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'

    }
	def __init__(self, output_activation, output_dims, n_dense_layers, dense_width, dense_dropout, max_timescale,n_positional_dims, phi_width):
		self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
		super().__init__()
		self.cnn_model = cnn_simple_model_temp()
		self.output_activation = output_activation
		self.output_dims = output_dims
		self.dense_layer = build_dense_dropout_model(n_dense_layers, dense_width, dense_dropout, self.dense_options)
		# self.demo_dense = build_dense_dropout_model(3,32,0,self.dense_options)
		self.dense_layer.add(Dense(output_dims, activation=output_activation))
		self.phi_width = phi_width
	def build(self, input_shapes):
		# pdb.set_trace()
		print(input_shapes)
		n_feature_maps = 64
		demo, times, values, measurements, loc, lengths= input_shapes
		self.resnet_model = self.cnn_model.build_resnet(((None,values[-1]), (None,values[-1])))
		n_samples = demo[0]
		n_chan = values[-1]
		self.n_values = values[1]
		
		self.demo_encoder = tf.keras.Sequential(
		    [
		        tf.keras.layers.Dense(self.phi_width, activation='relu'),
		        # tf.keras.layers.Dense(phi_input_dim)
		    ],
		    name='demo_encoder'
		)
		cnn_output_shape = (None,128+self.phi_width)
		self.dense_layer.build(cnn_output_shape)

	def call(self, inputs):

		self.demo, self.times, self.values, self.measurements, self.loc, lengths = inputs
		# pdb.set_trace()
		output = self.resnet_model((self.values, self.loc))
		demo_encoded = self.demo_encoder(self.demo)
		aggregated_values = tf.concat([output,demo_encoded], -1)
		dense_output = self.dense_layer(aggregated_values)
		return dense_output


	@classmethod
	def get_hyperparameters(cls):
		import tensorboard.plugins.hparams.api as hp
		from ..training_utils import HParamWithDefault
		return [
		    HParamWithDefault(
		        'n_dense_layers',
		        hp.Discrete([0,1,2,3,4,5]),
		        default=3
		    ),
		    HParamWithDefault(
		        'dense_width',
		        hp.Discrete([32,64,128,256,512]),
		        default=512
		    ),
		    HParamWithDefault(
		        'dense_dropout',
		        hp.Discrete([0.0,0.1,0.2,0.3]),
		        default=0.0
		    ),
		    HParamWithDefault(
		        'max_timescale',
		        hp.Discrete([10., 100., 1000.]),
		        default=100.
		    ),
		    HParamWithDefault(
                'n_positional_dims',
                hp.Discrete([64,128,256,512]),
                default=64
            ),
            HParamWithDefault(
                'phi_width',
                hp.Discrete([16, 32, 64, 128, 256, 512]),
                default=64
            )
		    ]

	@classmethod
	def from_hyperparameter_dict(cls, task, hparams):
		return cls(
		    output_activation=task.output_activation,
		    output_dims=task.n_outputs,
		    n_dense_layers=hparams['n_dense_layers'],
		    dense_width=hparams['dense_width'],
		    dense_dropout=hparams['dense_dropout'],
		    max_timescale=hparams['max_timescale'],
		    n_positional_dims=hparams['n_positional_dims'],
		    phi_width=hparams['phi_width']
		)

	@classmethod
	def from_config(cls, config):
	    return cls(**config)

	def get_config(self):
	    return self._config

	def data_preprocessing_fn(self):

		def flatten_unaligned_measurements(ts, labels):
			demo, X, Y, measurements, lengths = ts
			# Y = tf.expand_dims(Y,0)
			# Y = tf.keras.layers.ZeroPadding1D(padding=(1,0))(Y)
			Y0 = tf.transpose(Y)
			cast_meas_float = tf.cast(measurements, dtype=tf.float32)
			cast_meas_int = tf.cast(measurements, dtype=tf.int32)
			Y1 = Y0*tf.transpose(cast_meas_float)
			# pdb.set_trace()
			inds = tf.argsort(-tf.transpose(cast_meas_int))

			# tf.print(inds[:,0], tf.argsort(-cast_meas_int[:,0]))

			Y = tf.gather(Y1, inds, batch_dims=1)
			Y = tf.pad(Y, tf.constant([[0,0],[1,0]]), "CONSTANT")
			ids = tf.transpose(tf.cumsum(tf.cast(measurements, dtype=tf.int32),-2))

			# ids = tf.cast(ids[:,:,0], dtype=tf.int32)
			Y = tf.transpose(tf.gather(Y, ids,batch_dims=1))
			# pdb.set_trace()
			return (demo, X, Y, cast_meas_float, tf.ones_like(cast_meas_float), lengths), labels
		return flatten_unaligned_measurements
	    
	@classmethod
	def get_default(cls, task):
	    hyperparams = cls.get_hyperparameters()
	    return cls.from_hyperparameter_dict(
	        task,
	        {
	            h.name: h._default for h in hyperparams
	        }
	    )

