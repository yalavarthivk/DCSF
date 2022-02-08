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
import tensorflow_probability as tfp
get_output_shapes = tf.compat.v1.data.get_output_shapes
get_output_types = tf.compat.v1.data.get_output_types
make_one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator
from .cnn_model import cnn_act_simple_model as cnn_model
from tensorflow.keras.layers import Dense, Conv1D, Activation
# from keras_transformer.attention import MultiHeadSelfAttention
from .preproc import data_processing
from .set_utils import (
    build_dense_dropout_model, PaddedToSegments, SegmentAggregation,
    cumulative_softmax_weighting, cumulative_segment_mean)
from .set_utils import build_dense_dropout_model_2
from .utils import segment_softmax
from itertools import chain
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os 
import sys



class Classifier_RESNET_act_att(tf.keras.Model):
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
		self.cts = data_processing()
		self.cnn_model = cnn_model(n_cnn_layers)
		self.to_segments = PaddedToSegments()
		self.dense_layer = build_dense_dropout_model_2(n_dense_layers, dense_width, dense_dropout, self.dense_options)
		# pdb.set_trace()
		self.max_timescale = max_timescale
		self.dense_layer.add(Conv1D(filters=output_dims[-1], kernel_size=1))
		self.dense_layer.add(Activation(output_activation))
		self._n_modalities = None
		self.return_sequences = False
		self.aggregation = SegmentAggregation(
            aggregation_fn='sum',
            cumulative=self.return_sequences
        )
	def build(self, input_shapes):
		print(input_shapes)
		n_feature_maps = 64
		demo, tt, times, values, measurements, lengths, demo_pose, lens, tim_pos= input_shapes

		tim = (times[0],None,1)


		n_chan = values[-1] 
		self.n_values = values[1]
		# pdb.set_trace()
		self.n_chan = n_chan + 1

		cnn_input = (None, self.n_chan)
		cnn_correct = (None,1)
		# pdb.set_trace()
		self.resnet_model = self.cnn_model.build_resnet((cnn_input,cnn_correct))
		cnn_output_shape = (None,128)
		# self.dense_layer.build((None, (self.n_positional_dims)))
		self.dense_layer.build((None, None,128))

	def call(self, inputs):

		self.demo,self.tt, self.times, self.values, self.measurements, lengths, demo_pose, lens, tim_pos = inputs
		collected_values, segment_val_ids = self.to_segments(self.values, lengths)
		collected_tim_pos, segment_tim_pos_ids = self.to_segments(tim_pos, lengths)
		# pdb.set_trace()
		collected_tt, segment_times = self.to_segments(self.tt, lengths)
		collected_values = tf.concat([collected_values,collected_tt],-1)
		cnn_val_out = self.resnet_model((collected_values, collected_tim_pos))
		collected_times, segment_times = self.to_segments(self.times, lengths)
		
		# pdb.set_trace()
		wts = tf.cumsum(collected_tim_pos, -2)
		cnn_out = tf.math.divide_no_nan(tf.cumsum(cnn_val_out, -2),wts)
		cnn_out = cnn_val_out
		cnn_out = cnn_out*tf.tile(collected_tim_pos, [1,1,128])
		cnn_out = tf.keras.layers.ZeroPadding1D(padding=(0,(tf.math.reduce_max(self.times)+1-cnn_val_out.shape[-2])))(cnn_out)
		collected_times = tf.cast(collected_times[:,:,0], dtype=tf.int32)
		c_t_p = tf.keras.layers.ZeroPadding1D(padding=(0,(tf.math.reduce_max(self.times)+1-cnn_val_out.shape[-2])))(collected_tim_pos)
		cnn_out = tf.keras.layers.ZeroPadding1D(padding=(1,0))(cnn_out)
		# pdb.set_trace()
		idx = tf.gather(c_t_p, collected_times, batch_dims=1)
		ids = tf.cumsum(idx,-2)
		ids = tf.cast(ids[:,:,0], dtype=tf.int32)
		cnn_out = tf.gather(cnn_out, ids, batch_dims=1)
		# cnn_out = tf.gather(cnn_out, collected_times, batch_dims=1)
		segment_ids1 = segment_val_ids
		# segment_ids1 = tf.concat([segment_val_ids, segment_demo_ids],0)
		aggregated_values = self.aggregation(cnn_out, segment_ids1)
		# aggregated_values = tf.concat([aggregated_values,demo_encoded], -1)
		dens_output = self.dense_layer(aggregated_values)
		return dens_output
		
	def tf_fill(cnn_out):
		mask = tf.cast(cnn_out, tf.bool)
		values = tf.concat([[math.nan], tf.boolean_mask(input, mask)], axis=0)

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
		        hp.Discrete([0.01,0.1,0.2,0.3]),
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
            	hp.Discrete([0,1,2,3]),
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
			# tf.print(tf.shape(X))
			# pdb.set_trace()
			t_p = tf.range(tf.shape(X)[0], dtype=tf.float32)

			if self._n_modalities is None:
				self._n_modalities = int(measurements.get_shape()[-1])
			p,q = tf.shape(measurements)[0], measurements.shape[1]
			
			
			cast_meas_float = tf.cast(measurements, dtype = tf.float32)
			cast_meas_int = tf.cast(measurements, dtype=tf.int32)

			meas_pos = tf.cast(tf.reduce_sum(cast_meas_int,0), dtype=tf.bool)
			meas_len = tf.reduce_sum(cast_meas_int,axis=0)
			demo_pos = tf.cast(demo, dtype=tf.bool)
			lens = tf.reduce_sum(cast_meas_int) + tf.reduce_sum(tf.cast(demo, dtype=tf.int32))
			# pdb.set_trace()
			X = tf.expand_dims(X,-1)
			t_p = tf.expand_dims(t_p, -1)
			X0 = tf.transpose(tf.tile(X, [1,q]))
			t_p = tf.transpose(tf.tile(t_p,[1,q]))
			# tf.print('f', Y[0])
			Y0 = tf.transpose(Y)

			X1 = X0*tf.transpose(cast_meas_float)
			# t_p = t_p*tf.transpose(cast_meas_float)
			Y1 = Y0*tf.transpose(cast_meas_float)
			
			inds = tf.argsort(-tf.transpose(cast_meas_int))

			# tf.print(inds[:,0], tf.argsort(-cast_meas_int[:,0]))
			X = tf.gather(X1, inds, batch_dims=1)
			Y = tf.gather(Y1, inds, batch_dims=1)
			t_p = tf.gather(t_p, inds, batch_dims=1)
			X = tf.expand_dims(X, -1)
			t_p = tf.expand_dims(t_p, -1)
			Y = tf.expand_dims(Y, -1)
			# tf.print(ts[1],X[0,:,0], Y[0,:,0], meas_len[0], meas_pos[0])
			csd = tf.range(1,q+1)
			# csd = csd*tf.cast(meas_pos, dtype=tf.int32)-1
			csd = tf.expand_dims(csd,1)
			csd = tf.tile(csd, [1,p])
			csd = csd*tf.transpose(cast_meas_int)-1
			csd = tf.gather(csd, inds, batch_dims=1)
			# tf.print(csd[3], meas_len[3])
			csd = tf.one_hot(csd, depth = self._n_modalities)
			
			val_outs = tf.concat([csd, Y], -1)
			val_outs = val_outs[:,0:tf.math.reduce_max(meas_len),:]

			demo_ind = tf.cast(tf.cast(demo, dtype=tf.bool), dtype=tf.int32)
			d_p = demo.shape[0]
			d_inds = tf.range(q, q+d_p, dtype=tf.int32)*demo_ind
			d_inds = tf.where(tf.equal(d_inds, 0), tf.zeros_like(d_inds)-1, d_inds)
			d_Y = tf.expand_dims(demo,1)
			d_csd = tf.one_hot(d_inds, depth = self._n_modalities)
			tim_pos = -tf.sort(-tf.cast(measurements, dtype=tf.float32),0)
			tim_pos = tf.expand_dims(tf.transpose(tim_pos),-1)
			tim_pos = tim_pos[:,0:tf.math.reduce_max(meas_len),:]
			# pdb.set_trace()
			X = X[:,0:tf.math.reduce_max(meas_len),:]
			# pdb.set_trace()
			# t_p = t_p[:,0:tf.math.reduce_max(meas_len),:]
			t_p = tf.argsort(tf.cast(t_p, dtype=tf.int32), -2)
			max_vals = tf.math.reduce_max(X, 1)
			max_vals = tf.tile(max_vals[:, None, :], (1, tf.shape(X)[1], 1))
			X = tf.math.divide_no_nan(X, max_vals)
			return (demo, X, t_p, val_outs, cast_meas_float, meas_pos, demo_pos, lens, tim_pos), labels
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

