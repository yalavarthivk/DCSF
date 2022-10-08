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
from tensorflow.keras.layers import Dense
from .preproc import data_processing
from .set_utils import (
    build_dense_dropout_model, PaddedToSegments, SegmentAggregation,
    cumulative_softmax_weighting, cumulative_segment_mean)
from .utils import segment_softmax
from itertools import chain
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os 
import sys

class cnn_model():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        
        n_feature_maps = 64
        input_layer = keras.Input(input_shapes)

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        
        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_2)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
 
        self.gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        model = keras.Model(input_layer, self.gap_layer)
        return model

class RESNET(tf.keras.Model):
	dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'

    }
	def __init__(self, output_activation, output_dims):
		self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
		super().__init__()
		self.cnn_model = cnn_model()
		self.output_activation = output_activation
		self.output_dims = output_dims
	def build(self, input_shapes):
		print(input_shapes)
		n_feature_maps = 64
		demo, times, values, measurements, lengths= input_shapes
		self.resnet_model = self.cnn_model.build_resnet((None, values[-1]))
		self.dense_layer = tf.keras.Sequential(
		    [
		        tf.keras.layers.Dense(self.output_dims, activation=self.output_activation),
		    ],
		    name='dense_layer'
		)
		n_samples = demo[0]
		n_chan = values[-1]
		self.n_values = values[1]
		cnn_output_shape = (None,128)
		self.dense_layer.build(cnn_output_shape)

	def call(self, inputs):

		self.demo, self.times, self.values, self.measurements, lengths = inputs
		# pdb.set_trace()
		output = self.resnet_model(self.values)
		dense_output = self.dense_layer(output)
		return dense_output


	@classmethod
	def get_hyperparameters(cls):
		import tensorboard.plugins.hparams.api as hp
		from ..training_utils import HParamWithDefault
		return []

	@classmethod
	def from_hyperparameter_dict(cls, task, hparams):
		return cls(
		    output_activation=task.output_activation,
		    output_dims=task.n_outputs
		)

	@classmethod
	def from_config(cls, config):
	    return cls(**config)

	def get_config(self):
	    return self._config

	def data_preprocessing_fn(self):

		def flatten_unaligned_measurements(ts, labels):
			demo, X, Y, measurements, lengths = ts
			Y = tf.concat([Y, tf.cast(measurements, dtype=tf.float32)], -1)
			return (demo, X, Y, measurements, lengths), labels
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

