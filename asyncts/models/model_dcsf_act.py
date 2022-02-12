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
from tensorflow.keras.layers import Dense, Conv1D, Activation
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

class cnn_model():
    def __init__(self, n_layers):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()
        self.n_layers = n_layers
    def build_resnet(self, input_shapes):
        # pdb.set_trace()
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(input_layer1)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(conv_x)
        
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(conv_y)
        conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps])
        # conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(input_layer1)
        shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps])
        # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps])

        # BLOCK 2

        for i in range(self.n_layers):

            conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            # 
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_x)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_y)
            conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps*2])

            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)

            output_block_1 = output_block_2
        if self.n_layers == 0:
            output_block_1 = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            output_block_1 = keras.layers.Activation('relu')(output_block_1)


        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps*2])
        model = keras.Model((input_layer, input_correction), output_block_1)
        return model

class DCSF_act(tf.keras.Model):
	dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'

    }
	def __init__(self, output_activation, output_dims, n_dense_layers, dense_width, dense_dropout, n_cnn_layers):
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
		self.n_chan = n_chan + 1

		cnn_input = (None, self.n_chan)
		cnn_correct = (None,1)
		# pdb.set_trace()
		self.resnet_model = self.cnn_model.build_resnet((cnn_input,cnn_correct))
		cnn_output_shape = (None,128)
		self.dense_layer.build((None, None,128))

	def call(self, inputs):

		self.demo,self.tt, self.times, self.values, self.measurements, lengths, demo_pose, lens, tim_pos = inputs
		collected_values, segment_val_ids = self.to_segments(self.values, lengths)
		collected_tim_pos, segment_tim_pos_ids = self.to_segments(tim_pos, lengths)
		collected_tt, segment_times = self.to_segments(self.tt, lengths)
		collected_values = tf.concat([collected_values,collected_tt],-1)
		cnn_val_out = self.resnet_model((collected_values, collected_tim_pos))
		collected_times, segment_times = self.to_segments(self.times, lengths)
		
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
		aggregated_values = self.aggregation(cnn_out, segment_val_ids)
		dens_output = self.dense_layer(aggregated_values)
		return dens_output
		

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

			X = tf.expand_dims(X,-1)
			t_p = tf.expand_dims(t_p, -1)
			X0 = tf.transpose(tf.tile(X, [1,q]))
			t_p = tf.transpose(tf.tile(t_p,[1,q]))

			Y0 = tf.transpose(Y)

			X1 = X0*tf.transpose(cast_meas_float)

			Y1 = Y0*tf.transpose(cast_meas_float)
			
			inds = tf.argsort(-tf.transpose(cast_meas_int))

			X = tf.gather(X1, inds, batch_dims=1)
			Y = tf.gather(Y1, inds, batch_dims=1)
			t_p = tf.gather(t_p, inds, batch_dims=1)
			X = tf.expand_dims(X, -1)
			t_p = tf.expand_dims(t_p, -1)
			Y = tf.expand_dims(Y, -1)
			csd = tf.range(1,q+1)
			csd = tf.expand_dims(csd,1)
			csd = tf.tile(csd, [1,p])
			csd = csd*tf.transpose(cast_meas_int)-1
			csd = tf.gather(csd, inds, batch_dims=1)
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
			X = X[:,0:tf.math.reduce_max(meas_len),:]
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

