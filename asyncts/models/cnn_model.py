import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

class dense_layer(tf.keras.layers.Layer):

    def __init__(self, nb_classes):
        super().__init__()
        self.nb_classes = nb_classes

    def build(self, input_shape):
        self.input_shapes = input_shape[-1]

    def call(self, inputs):
        d1 = keras.layers.Dense(self.nb_classes, activation='softmax', input_shape=[inputs.shape[-1]])(inputs)
        return d1



class cnn_simple_model_temp():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        
        n_feature_maps = 64
        input_layer1 = keras.Input(input_shapes[0])
        input_correction = keras.Input(input_shapes[1])
        input_layer = input_layer1*input_correction
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
        wts = tf.math.reduce_sum(tf.tile(input_correction[:,:,0:1], [1,1, n_feature_maps*2]), -2)
        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        output_block_3 = output_block_3*tf.tile(input_correction[:,:,0:1], [1,1, n_feature_maps*2])
        self.gap_layer = tf.math.reduce_sum(output_block_3, -2)/wts
        # self.gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        model = keras.Model((input_layer1, input_correction), self.gap_layer)
        return model



class cnn_simple_model():
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
        
class cnn_model_att():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='same')(input_layer1)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='same')(conv_y)
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same')(input_layer1)
        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # BLOCK 2
        time_pos_2 = time_pos*tf.tile(input_correction, [1,1,pos_enc])
        output_block_enc = tf.concat([output_block_1, time_pos_2], -1)

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_enc)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_enc)
        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_2)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        output_block_3 = output_block_3*tf.tile(input_correction, [1,1,n_feature_maps*2])
        
        # query = keras.layers.Conv1D(filters=pos_enc, kernel_size=1, padding='same')(output_block_3) + time_pos_2
        # pdb.set_trace()

        # att_layer = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, attention_axes=-2)(query, query)
        # att_layer_1 = keras.layers.Conv1D(filters=pos_enc, kernel_size=1,  padding='same')(att_layer)
        # att_layer_1 = tf.keras.layers.LayerNormalization()(att_layer_1)
        # att_layer_1 = keras.layers.Activation('relu')(att_layer_1)
        # att_layer = keras.layers.add([att_layer_1, att_layer])


        # att_layer = att_layer*tf.tile(input_correction, [1,1,pos_enc])
        att_layer = output_block_3
        weight = tf.reduce_sum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        self.gap_layer = tf.reduce_sum(att_layer, -2)/weight
        model = keras.Model((input_layer,time_pos, input_correction), self.gap_layer)
        return model

class cnn_model_att_gap():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='same')(input_layer1)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='same')(conv_y)
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same')(input_layer1)
        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_1)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_1)
        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_2)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        output_block_3 = output_block_3*tf.tile(input_correction, [1,1,n_feature_maps*2])
        pool_layer = tf.reduce_sum(output_block_3, -2)
        weight = tf.reduce_sum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        gap_layer = pool_layer/weight
         
        time_pos_2 = time_pos*tf.tile(input_correction, [1,1,pos_enc])
        query = keras.layers.Conv1D(filters=pos_enc, kernel_size=1, padding='same')(output_block_3) + time_pos_2
        # pdb.set_trace()
        mask = tf.squeeze(input_correction, -1)

        att_layer = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, attention_axes=-2)(query, query)
        att_layer = tf.reduce_sum(att_layer, -2)
        self.cat_layer = tf.concat([att_layer, gap_layer], -1)
        model = keras.Model((input_layer,time_pos, input_correction), self.cat_layer)
        return model


class cnn_model_act():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])
        # pdb.set_trace()
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='causal')(input_layer1)
        # conv_x = keras.layers.LayerNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='causal')(conv_x)
        
        # conv_y = keras.layers.LayerNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='causal')(conv_y)
        conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps])
        # conv_z = keras.layers.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(input_layer1)
        shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps])
        # shortcut_y = keras.layers.LayerNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # pdb.set_trace()
        
        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps])
        # time_pos_2 = time_pos*tf.tile(input_correction, [1,1,pos_enc])
        # output_block_enc = tf.concat([output_block_1, time_pos], -1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='causal')(output_block_1)
        # 
        # conv_x = keras.layers.LayerNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps*2])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='causal')(conv_x)
        # 
        # conv_y = keras.layers.LayerNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='causal')(conv_y)
        conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps*2])
        # conv_z = keras.layers.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
        shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps*2])
        # shortcut_y = keras.layers.LayerNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        # output_block_2 = output_block_2*tf.tile(input_correction, [1,1,n_feature_maps*2])

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='causal')(output_block_2)
        # 
        # conv_x = keras.layers.LayerNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps*2])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='causal')(conv_x)
        # 
        # conv_y = keras.layers.LayerNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='causal')(conv_y)
        conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps*2])
        # conv_z = keras.layers.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_2)
        shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps*2])
        # shortcut_y = keras.layers.LayerNormalization()(shortcut_y)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        output_block_3 = output_block_3*tf.tile(input_correction, [1,1,n_feature_maps*2])

        # pool_layer = tf.cumsum(output_block_3, -2)
        # weight = tf.cumsum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        
        # gap_layer_2 = pool_layer/weight
        model = keras.Model((input_layer,time_pos, input_correction), output_block_3)
        return model
class cnn_act_simple_model():
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
            # conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_x)
            # 
            # conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_y)
            conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps*2])
            # conv_z = keras.layers.BatchNormalization()(conv_z)

            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps*2])
            # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)

            output_block_1 = output_block_2
        if self.n_layers == 0:
            output_block_1 = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            output_block_1 = keras.layers.Activation('relu')(output_block_1)


        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps*2])
        # pdb.set_trace()
        model = keras.Model((input_layer, input_correction), output_block_1)
        return model

class cnn_act_dns_model():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        
        n_feature_maps = 64
        input_layer = keras.Input(input_shapes)

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same',input_shape=input_shapes[1:])(input_layer)
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
 
        model = keras.Model(input_layer, output_block_3)
        return model
class cnn_model_act2():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,1,input_shape1[-1]])
        # pdb.set_trace()
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='same',input_shape=input_shape1[1:])(input_layer1)
        # conv_x = keras.layers.LayerNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,1,n_feature_maps])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='same')(conv_x)
        
        # conv_y = keras.layers.LayerNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,1,n_feature_maps])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='same')(conv_y)
        conv_z = conv_z*tf.tile(input_correction, [1,1,1,n_feature_maps])
        # conv_z = keras.layers.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same')(input_layer1)
        shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,1,n_feature_maps])
        # shortcut_y = keras.layers.LayerNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # pdb.set_trace()
        
        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,1,n_feature_maps])
        # time_pos_2 = time_pos*tf.tile(input_correction, [1,1,pos_enc])
        # output_block_enc = tf.concat([output_block_1, time_pos], -1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_1)
        # 
        # conv_x = keras.layers.LayerNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,1,n_feature_maps*2])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        # 
        # conv_y = keras.layers.LayerNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,1,n_feature_maps*2])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        conv_z = conv_z*tf.tile(input_correction, [1,1,1,n_feature_maps*2])
        # conv_z = keras.layers.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_1)
        shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,1,n_feature_maps*2])
        # shortcut_y = keras.layers.LayerNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        # output_block_2 = output_block_2*tf.tile(input_correction, [1,1,n_feature_maps*2])

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_2)
        # 
        # conv_x = keras.layers.LayerNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = conv_x*tf.tile(input_correction, [1,1,1,n_feature_maps*2])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
        # 
        # conv_y = keras.layers.LayerNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = conv_y*tf.tile(input_correction, [1,1,1,n_feature_maps*2])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
        conv_z = conv_z*tf.tile(input_correction, [1,1,1,n_feature_maps*2])
        # conv_z = keras.layers.LayerNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_2)
        shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,1,n_feature_maps*2])
        # shortcut_y = keras.layers.LayerNormalization()(shortcut_y)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        output_block_3 = output_block_3*tf.tile(input_correction, [1,1,1,n_feature_maps*2])

        # pool_layer = tf.cumsum(output_block_3, -2)
        # weight = tf.cumsum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        
        # gap_layer_2 = pool_layer/weight
        model = keras.Model((input_layer,time_pos, input_correction), output_block_3)
        return model

class cnn_model_temp():
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='same')(input_layer1)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        # conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='same')(conv_x)
        
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        # conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps])

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='same')(conv_y)
        # conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps])
        # conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same')(input_layer1)
        # shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps])
        # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps])

        # BLOCK 2

        for i in range(self.n_layers):

            conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_1)
            # 
            # conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            # conv_x = conv_x*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
            # 
            # conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            # conv_y = conv_y*tf.tile(input_correction, [1,1,n_feature_maps*2])

            conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
            # conv_z = conv_z*tf.tile(input_correction, [1,1,n_feature_maps*2])
            # conv_z = keras.layers.BatchNormalization()(conv_z)

            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_1)
            # shortcut_y = shortcut_y*tf.tile(input_correction, [1,1,n_feature_maps*2])
            # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)

            output_block_1 = output_block_2
        if self.n_layers == 0:
            output_block_1 = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # output_block_2 = output_block_2*tf.tile(input_correction, [1,1,n_feature_maps*2])

        # BLOCK 3


        output_block_1 = output_block_1*tf.tile(input_correction, [1,1,n_feature_maps*2])
        
        pool_layer = tf.reduce_sum(output_block_1, -2)
        weight = tf.reduce_sum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        gap_layer_2 = tf.math.divide_no_nan(pool_layer,weight)
        self.gap_layer = gap_layer_2
        model = keras.Model((input_layer,time_pos, input_correction), self.gap_layer)
        return model

class cnn_model_att_temp():
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        pos_enc = input_shapes[1][-1]
        input_shape1 = input_shapes[0]
        input_shape2 = input_shapes[2]
        input_shape_tim = input_shapes[1]
        n_feature_maps = 64
        input_layer = keras.Input(input_shape1)
        time_pos = keras.Input(input_shape_tim)
        input_correction = keras.Input(input_shape2)
 
        input_layer1 = input_layer*tf.tile(input_correction, [1,1,input_shape1[-1]])

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8,  padding='same')(input_layer1)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5,  padding='same')(conv_x)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3,  padding='same')(conv_y)
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='same')(input_layer1)
        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # BLOCK 2
        time_pos_2 = time_pos*tf.tile(input_correction, [1,1,pos_enc])
        output_block_enc = tf.concat([output_block_1, time_pos_2], -1)
        for i in range(self.n_layers):
            conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8,  padding='same')(output_block_enc)
            conv_x = keras.layers.Activation('relu')(conv_x)
            conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5,  padding='same')(conv_x)
            conv_y = keras.layers.Activation('relu')(conv_y)
            conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3,  padding='same')(conv_y)
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='same')(output_block_enc)
            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)
            output_block_enc = output_block_2
        if self.n_layers == 0:
            output_block_1 = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
            output_block_enc = keras.layers.Activation('relu')(output_block_1)

        output_block_enc = output_block_enc*tf.tile(input_correction, [1,1,n_feature_maps*2])

        
        att_layer = output_block_enc
        weight = tf.reduce_sum(tf.tile(input_correction, [1,1,n_feature_maps*2]), -2)
        self.gap_layer = tf.reduce_sum(att_layer, -2)/weight
        model = keras.Model((input_layer,time_pos, input_correction), self.gap_layer)
        return model
class resnet_act_simple():
    def __init__(self):
        super().__init__()
        # self.sin_act = sin_activation()
        # self.sin_act.build()

    def build_resnet(self, input_shapes):
        
        n_feature_maps = 64
        input_layer = keras.Input(input_shapes)

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(input_layer)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(conv_x)
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(conv_y)
        # conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1,  padding='causal')(input_layer)
        # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_x)
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_y)
        # conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_1)
        # shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        
        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_2)
        # conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_x)
        # conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(conv_y)
        # conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1,  padding='causal')(output_block_2)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        outs = tf.cumsum(output_block_3, -2)
        wts = tf.cumsum(tf.ones_like(outs), -2)
        self.gap_layer = outs/wts
        # self.gap_layer = output_block_3
        model = keras.Model(input_layer, self.gap_layer)
        return model