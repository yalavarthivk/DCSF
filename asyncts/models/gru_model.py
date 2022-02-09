import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

import pdb



class gru_model_temp():
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_gru(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        mask = (1 - tf.tile(input_correction, [1,1,input_shape1[-1]]))*10000
        # input_layer1 = input_layer1 + mask
        # pdb.set_trace()
        input_layer1 = tf.keras.layers.Masking(mask_value=10000, input_shape=input_shape1)(input_layer1)

        conv_x = keras.layers.GRU(n_feature_maps, return_sequences=True)(input_layer1)
        conv_x = keras.layers.Activation('relu')(conv_x)
        # conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_y = keras.layers.GRU(n_feature_maps, return_sequences=True)(conv_x)
        
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        # conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_z = keras.layers.GRU(n_feature_maps, return_sequences=True)(conv_y)
        # conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps])
        # conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y =  keras.layers.GRU(n_feature_maps, return_sequences=True)(input_layer1)
        # shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps])
        # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps])

        # BLOCK 2

        for i in range(self.n_layers):

            conv_x = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(output_block_1)
            # 
            # conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            # conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_y = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(conv_x)
            # 
            # conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            # conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_z = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(conv_y)
            # conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps*2])
            # conv_z = keras.layers.BatchNormalization()(conv_z)

            # expand channels for the sum
            shortcut_y = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(output_block_1)
            # shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps*2])
            # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)

            output_block_1 = output_block_2


        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps*2])
        
        pool_layer = tf.reduce_sum(output_block_1, -2)
        weight = tf.reduce_sum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        gap_layer_2 = pool_layer/weight
        self.gap_layer = gap_layer_2
        model = keras.Model((input_layer,time_pos, input_correction), self.gap_layer)
        return model

class gru_model_att_temp():
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_gru(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        conv_x = keras.layers.GRU(n_feature_maps, return_sequences=True)(input_layer1)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.GRU(n_feature_maps, return_sequences=True)(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.GRU(n_feature_maps, return_sequences=True)(conv_y)
        shortcut_y = keras.layers.GRU(n_feature_maps, return_sequences=True)(input_layer1)
        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # BLOCK 2
        time_pos_2 = time_pos*tf.tile(input_correction, [1,1,pos_enc])
        output_block_enc = tf.concat([output_block_1, time_pos_2], -1)
        for i in range(self.n_layers):
            conv_x = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(output_block_enc)
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_y = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(conv_x)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_z = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(conv_y)
            shortcut_y = keras.layers.GRU(n_feature_maps*2, return_sequences=True)(output_block_enc)
            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)
            output_block_enc = output_block_2

        output_block_enc = output_block_enc*tf.tile(input_correction, [1,1,n_feature_maps*2])

        
        att_layer = output_block_enc
        weight = tf.reduce_sum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        self.gap_layer = tf.reduce_sum(att_layer, -2)/weight
        model = keras.Model((input_layer,time_pos, input_correction), self.gap_layer)
        return model