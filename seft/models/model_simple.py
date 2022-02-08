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
# from .cnn_model import cnn_model
from .cnn_model import dense_layer
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


class cnn_model():
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers

    def build_resnet(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        gap_wts = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        gap_wts_layer = keras.Input(gap_wts)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='same')(input_layer1)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='same')(conv_y)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same')(input_layer1)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps])

        # BLOCK 2

        for i in range(self.n_layers):

            conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_1)
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
            
            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_1)
            
            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)
            output_block_2 = output_block_2*tf.tile(input_correction, [1,1,n_feature_maps*2])

            output_block_1 = output_block_2
        if self.n_layers == 0:
            output_block_1 = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            output_block_1 = keras.layers.Activation('relu')(output_block_1)
            output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps*2])

        self.gap_layer = output_block_1*tf.tile(gap_wts_layer, (1,1,n_feature_maps*2))
        self.gap_layer = tf.math.reduce_sum(self.gap_layer, 1)
        model = keras.Model((input_layer,gap_wts_layer, input_correction), self.gap_layer)
        return model


class simple(tf.keras.Model):
	dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'

    }
	def __init__(self, output_activation, output_dims, n_dense_layers, dense_width, dense_dropout, max_timescale,n_positional_dims, phi_width, n_cnn_layers):
		self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
		super().__init__()
		self.n_cnn_layers = n_cnn_layers
		self.cts = data_processing()
		self.cnn_model = cnn_model(self.n_cnn_layers)
		# self.att_layer = attention(dot_prod_dim, n_heads)
		self.to_segments = PaddedToSegments()
		self.dense_layer = build_dense_dropout_model(n_dense_layers, dense_width, dense_dropout, self.dense_options)
		# self.demo_dense = build_dense_dropout_model(3,32,0,self.dense_options)
		self.dense_layer.add(Dense(output_dims, activation=output_activation))
		self._n_modalities = None
		self.return_sequences = False
		self.max_timescale = max_timescale
		self.phi_width = phi_width
		self.aggregation = SegmentAggregation(
            aggregation_fn='sum',
            cumulative=self.return_sequences
        )

		# self.latent_width = latent_width
	def build(self, input_shapes):
		print(input_shapes)
		demo, X, values, gathered_avgps, gathered_inds = input_shapes

		self.demo_encoder = tf.keras.Sequential(
		    [
		        tf.keras.layers.Dense(self.phi_width, activation='relu'),
		        # tf.keras.layers.Dense(phi_input_dim)
		    ],
		    name='demo_encoder'
		)
		
		n_samples = demo[0]
		self.n_chan = values[-1]
		cnn_input = (None, self.n_chan)
		cnn_correct = (None,1)
		self.resnet_model = self.cnn_model.build_resnet((cnn_input,cnn_correct, cnn_correct))
		cnn_output_shape = (None,128)
		self.demo_encoder.build(demo) 
		self.dense_layer.build((None, (128+self.phi_width)))

	def call(self, inputs):

		self.demo, self.X, self.values, self.avgps, self.inds = inputs
		# pdb.set_trace()

		self.batch_siz = tf.shape(self.demo)[0]
		cnn_out = self.resnet_model((self.values, self.avgps[:,:,None], self.inds[:,:,None]))
		demo_encoded = self.demo_encoder(self.demo)

		aggregated_values = tf.concat([cnn_out,demo_encoded], -1)
		dens_output = self.dense_layer(aggregated_values)
		return dens_output
		
	def cnn_outs(self,data):
		csd = tf.expand_dims(self.measurements,-1)
		csd = tf.tile(csd,[1,1,1,data.shape[-1]])
		return tf.math.reduce_sum(tf.multiply(csd,data), -2)

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
            ),
            HParamWithDefault(
            	'n_cnn_layers',
            	hp.Discrete([1,2,3]),
            	default=2
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
		    phi_width=hparams['phi_width'],
		    n_cnn_layers=hparams['n_cnn_layers']
		)

	@classmethod
	def from_config(cls, config):
	    return cls(**config)

	def get_config(self):
	    return self._config

	def data_preprocessing_fn(self):

		def flatten_unaligned_measurements(ts, labels):
			demo, X, Y, measurements, lengths = ts
			tims = X


			if self._n_modalities is None:
				self._n_modalities = int(measurements.get_shape()[-1])
			p,q = tf.shape(measurements)[0], measurements.shape[1]
			
			
			cast_meas_float = tf.cast(measurements, dtype = tf.float32)
			cast_meas_int = tf.cast(measurements, dtype=tf.int32)

			meas_len = tf.reduce_sum(cast_meas_float,axis=0)
			
			padds = tf.cast(tf.cast(meas_len,dtype=tf.bool), tf.int32)
			padds = tf.tile(padds[None, :], (8,1))
			
			wts_avgp = tf.tile(meas_len[None,:], (p, 1))
			wts_avgp = tf.concat([wts_avgp, tf.cast(0*padds, dtype=tf.float32)], 0)
			wts_avgp = tf.math.divide_no_nan(1., wts_avgp)

			vals_pad = tf.concat([Y, tf.cast(0*padds, dtype=tf.float32)], 0)

			channs_pad = tf.tile(tf.range(1, q+1)[None,:], (tf.shape(vals_pad)[0],1))

			mask = tf.concat([cast_meas_int, padds], 0)
			mask_positions = tf.cast(tf.where(tf.transpose(mask)), tf.int32)

			gathered_channs =  tf.gather_nd(tf.transpose(channs_pad), mask_positions)

			gathered_vals = tf.gather_nd(tf.transpose(vals_pad), mask_positions)

			gathered_channs = tf.one_hot(gathered_channs, depth = self._n_modalities)

			values = tf.concat([gathered_channs, gathered_vals[:,None]], -1)
			gathered_avgps = tf.gather_nd(tf.transpose(wts_avgp), mask_positions)

			gathered_inds = tf.cast(tf.cast(gathered_avgps, dtype=tf.bool), tf.float32)

# demo_outs
			return (demo, X, values, gathered_avgps, gathered_inds), labels
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
