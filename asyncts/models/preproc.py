import tensorflow as tf
import pdb
import numpy as np
from collections.abc import Sequence
import time
from tensorflow import keras 

class data_processing(keras.layers.Layer):
	def __init__(self):
		self.output_temp = []
		super().__init__()

	def build(self, input_shape):
		self.demo, self.times, self.values, self.measurements = input_shape
		self.depth = self.demo[-1] + self.values[-1]
		self.val_chan = self.values[-1]
		self.demo_chan = self.demo[-1]

	def call(self, inputs):
		self.demo, self.times, self.values, self.measurements = inputs
		def body1(x,z):
			
			asd = tf.where(self.measurements[:,:,x]==True)
			asd_zero = tf.where(self.measurements[:,:,x]==False)

			no_val_tim = tf.math.reduce_sum(self.meas[:,:,x],axis=1)
			no_val_tim = tf.expand_dims(no_val_tim,-1)
			asd_zero = tf.where(no_val_tim==0)
			
			pad_zero = tf.zeros_like(asd_zero, dtype=tf.float32)
			# pad_zero = tf.zeros([asd_zero.shape[0],1])
			asd_2 = tf.concat([asd,asd_zero],0)
			
			# pdb.set_trace()

			val_tim = self.times[self.measurements[:,:,x]]
			val_val = self.values[:,:,x][self.measurements[:,:,x]]
			val_csd = tf.ones_like(val_tim,dtype='int32')*x
			

			val_tim2 = tf.concat([val_tim, pad_zero[:,0]],0)
			val_val2 = tf.concat([val_val, pad_zero[:,0]],0)
			val_csd2 = tf.concat([val_csd, tf.cast((pad_zero[:,0]-1),'int32')],0)
			

			if asd_zero.shape[0] != 0:
				sort_order = tf.argsort(asd_2[:,0],axis=0)
				asd_new = tf.gather(asd_2[:,0], sort_order)
				val_tim2 = tf.gather(val_tim2, sort_order)
				val_val2 = tf.gather(val_val2, sort_order)
				val_csd2 = tf.gather(val_csd2, sort_order)
			else:
				asd_new = asd_2[:,0]


			tsd = tf.RaggedTensor.from_value_rowids(val_tim2,asd_new)
			
			vsd = tf.RaggedTensor.from_value_rowids(val_val2,asd_new)
			
			csd = tf.RaggedTensor.from_value_rowids(val_csd2,asd_new)
			pdb.set_trace()
			tsd = tf.expand_dims(tsd,-1)
			vsd = tf.expand_dims(vsd,-1)
			csd = tf.one_hot(csd,depth=self.depth)

			
			self.output_temp = []

			temp = tf.concat([tsd,csd,vsd], axis=-1)

			self.output_data.append(temp)
			return (tf.add(x,1),z)

		def demos(x,z):
			# pdb.set_trace()
			asd = self.demo[:,x:x+1]
			csd = tf.one_hot(tf.ones([self.demo.shape[0],1],dtype='int32')*(self.values.shape[-1]+x-1), self.depth)
			tim = tf.zeros([self.demo.shape[0],1,1])
			val = tf.expand_dims(asd,-1)
			temp = tf.RaggedTensor.from_tensor(tf.concat([tim,csd,val],axis=-1))
			self.output_data.append(temp)
			return (tf.add(x,1),z)
		def c1(x,z):
			return x < self.val_chan
		def c2(x,z):
			return x < self.demo_chan

		self.output_data = []
		# pdb.set_trace()
		self.output_temp = []
		self.meas = tf.cast(self.measurements,dtype='int32')
		
		x = tf.constant(0)
		z = tf.constant(0)
		pdb.set_trace()
		result0 = tf.while_loop(c1, body1, ([x,z]))
		result1 = tf.while_loop(c2, demos, ([x,z]))
		
		return self.output_data
	def compute_output_shape(self, input_shape):
		return (input_shape[0][0], None, self.depth+2)