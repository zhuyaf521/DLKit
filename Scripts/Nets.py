from itertools import chain
import tensorflow as tf
from tensorflow.keras import layers, Sequential, losses, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import matplotlib as mpl
from tensorflow.keras import backend as K

# 设置 GPU 显存使用方式
# 获取 GPU 设备列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 为增长式占用
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)
mpl.use('Agg')


class CNN(object):
    def __init__(self, input_row, input_col):
        super(CNN, self).__init__()
        self.input_row = input_row
        self.input_col = input_col
        self.network = None

        self.network = self.convolutional_network()

    def convolutional_network(self):
        if self.input_col:
            inputs = tf.keras.Input(shape=(self.input_row, self.input_col))
            x = layers.Conv1D(128, kernel_size=3)(inputs)
        else:
            inputs = tf.keras.Input(shape=self.input_row)
            x = Embedding(input_dim=22, output_dim=5, input_length=self.input_row)(inputs)
            x = layers.Conv1D(128, kernel_size=3)(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool1D(pool_size=2, strides=1, padding='same')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Conv1D(128, kernel_size=3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool1D(pool_size=2, strides=1, padding='same')(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
        DeepSUMO = Model(inputs=[inputs], outputs=[output], name="DeepSUMO")
        DeepSUMO.compile(optimizer=optimizers.Adam(),
                         loss='binary_crossentropy',
                         metrics=['accuracy'],
                         experimental_run_tf_function=False)
        return DeepSUMO

    def get_network(self):
        return self.network


def res_net_block(input_data, filters, strides=1):
    x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if strides != 1:
        down_sample = layers.Conv1D(filters, kernel_size=1, strides=strides)(input_data)
    else:
        down_sample = input_data
    x = layers.Add()([x, down_sample])
    output = layers.Activation('relu')(x)

    return output


class RSCNN(object):
    def __init__(self, input_row, input_col):
        super(RSCNN, self).__init__()
        self.input_row = input_row
        self.input_col = input_col
        self.network = None

        self.network = self.convolutional_network()

    def convolutional_network(self):
        if self.input_col:
            inputs = tf.keras.Input(shape=(self.input_row, self.input_col))
            x = layers.Conv1D(128, kernel_size=1)(inputs)
        else:
            inputs = tf.keras.Input(shape=self.input_row)
            x = Embedding(input_dim=22, output_dim=5, input_length=self.input_row)(inputs)
            x = layers.Conv1D(128, kernel_size=1)(x)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool1D(pool_size=2, strides=1, padding='same')(x)
        x = layers.Dropout(0.5)(x)

        x = res_net_block(x, 128)
        x = layers.MaxPool1D(2)(x)
        x = layers.Dropout(0.5)(x)

        x = res_net_block(x, 128)
        x = layers.MaxPool1D(2)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
        DeepSUMO = Model(inputs=[inputs], outputs=[output], name="DeepSUMO")
        DeepSUMO.compile(optimizer=optimizers.Adam(),
                         loss='binary_crossentropy',
                         metrics=['accuracy'],
                         experimental_run_tf_function=False)
        return DeepSUMO

    def get_network(self):
        return self.network


class BiLSTM(object):
    def __init__(self, input_row, input_col):
        super(BiLSTM, self).__init__()
        self.input_row = input_row
        self.input_col = input_col
        self.network = None

        self.network = self.recurrent_network()

    def recurrent_network(self):
        if self.input_col:
            inputs = tf.keras.Input(shape=(self.input_row, self.input_col))
            x = layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(inputs)
        else:
            inputs = tf.keras.Input(shape=self.input_row)
            x = Embedding(input_dim=22, output_dim=5, input_length=self.input_row)(inputs)
            x = layers.Bidirectional(
                layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(x)

        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
        DeepSUMO = Model(inputs=[inputs], outputs=[output], name="DeepSUMO")
        DeepSUMO.compile(optimizer=optimizers.Adam(),
                         loss='binary_crossentropy',
                         metrics=['accuracy'],
                         experimental_run_tf_function=False)
        return DeepSUMO

    def get_network(self):
        return self.network


class BiGRU(object):
    def __init__(self, input_row, input_col):
        super(BiGRU, self).__init__()
        self.input_row = input_row
        self.input_col = input_col
        self.network = None

        self.network = self.recurrent_network()

    def recurrent_network(self):
        if self.input_col:
            inputs = tf.keras.Input(shape=(self.input_row, self.input_col))
            x = layers.Bidirectional(
                layers.GRU(32, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(inputs)
        else:
            inputs = tf.keras.Input(shape=self.input_row)
            x = Embedding(input_dim=22, output_dim=5, input_length=self.input_row)(inputs)
            x = layers.Bidirectional(
                layers.GRU(32, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(x)

        x = layers.Bidirectional(
            layers.GRU(32, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
        DeepSUMO = Model(inputs=[inputs], outputs=[output], name="DeepSUMO")
        DeepSUMO.compile(optimizer=optimizers.Adam(),
                         loss='binary_crossentropy',
                         metrics=['accuracy'],
                         experimental_run_tf_function=False)
        return DeepSUMO

    def get_network(self):
        return self.network


def channel_attention_1D(input_feature, ratio=8, name=""):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_one_" + str(name))
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=False,
                             bias_initializer='zeros',
                             name="channel_attention_shared_two_" + str(name))

    avg_pool = GlobalAveragePooling1D()(input_feature)
    max_pool = GlobalMaxPooling1D()(input_feature)

    avg_pool = Reshape((1, channel))(avg_pool)
    max_pool = Reshape((1, channel))(max_pool)

    avg_pool = shared_layer_one(avg_pool)
    max_pool = shared_layer_one(max_pool)

    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    print(cbam_feature.shape)

    return multiply([input_feature, cbam_feature])


def spatial_attention_1D(input_feature, name=""):
    kernel_size = 7

    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=2)([avg_pool, max_pool])

    cbam_feature = Conv1D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False,
                          name="spatial_attention_" + str(name))(concat)
    cbam_feature = Activation('sigmoid')(cbam_feature)
    print(cbam_feature.shape)

    return multiply([input_feature, cbam_feature])


class HDeepSPred(object):
    def __init__(self, input_row, input_col):
        super(HDeepSPred, self).__init__()
        self.input_row = input_row
        self.input_col = input_col
        self.network = None

        self.network = self.recurrent_network()

    def recurrent_network(self):
        if self.input_col:
            inputs = tf.keras.Input(shape=(self.input_row, self.input_col))
            x = layers.Conv1D(128, kernel_size=1)(inputs)
            z = layers.Bidirectional(
                layers.GRU(32, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(inputs)
        else:
            inputs = tf.keras.Input(shape=self.input_row)
            y = Embedding(input_dim=22, output_dim=5, input_length=self.input_row)(inputs)
            x = layers.Conv1D(128, kernel_size=1)(y)
            z = layers.Bidirectional(
                layers.GRU(32, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(y)

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPool1D(pool_size=2, strides=1, padding='same')(x)
        x = layers.Dropout(0.5)(x)

        x = res_net_block(x, 128)
        x = layers.MaxPool1D(2)(x)
        x = layers.Dropout(0.5)(x)

        x = res_net_block(x, 128)
        x = layers.MaxPool1D(2)(x)
        x = layers.Dropout(0.5)(x)

        x = channel_attention_1D(x, name='ca1')

        z = layers.Bidirectional(
            layers.GRU(32, return_sequences=True, dropout=0.2, recurrent_initializer='orthogonal'))(z)

        z = spatial_attention_1D(z)

        x = layers.Flatten()(x)
        z = layers.Flatten()(z)

        merge = tf.concat([x, z], 1)

        merge = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(merge)
        merge = layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(merge)

        output = layers.Dense(1, activation=tf.nn.sigmoid)(merge)
        DeepSUMO = Model(inputs=[inputs], outputs=[output], name="DeepSUMO")
        DeepSUMO.compile(optimizer=optimizers.Adam(),
                         loss='binary_crossentropy',
                         metrics=['accuracy'],
                         experimental_run_tf_function=False)
        return DeepSUMO

    def get_network(self):
        return self.network


def conv_factory(x, filters, kernel_size, pool_size, dropout, padding):
    if padding == 0:
        filling = 'valid'
    else:
        filling = 'same'
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding=filling,
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if pool_size != 0:
        x = MaxPooling1D(pool_size=pool_size)(x)
    if dropout:
        x = Dropout(dropout)(x)
    return x


def dense_factory(x, units, dropout):
    x = Dense(units=units, activation='relu', kernel_initializer='uniform')(x)
    if dropout != 0:
        x = Dropout(dropout)(x)
    return x


def rnn_factory(x, net_type, param, cycle=False):
    if cycle:
        if net_type == 1:
            output = Bidirectional(LSTM(param[0], return_sequences=True, dropout=param[1]))(x)
        else:
            output = Bidirectional(GRU(param[0], return_sequences=True, dropout=param[1]))(x)
    else:
        if net_type == 1:
            output = LSTM(param[0], return_sequences=True, dropout=param[1])(
                x)
        else:
            output = GRU(param[0], return_sequences=True, dropout=param[1])(x)

    return output


class CnnNetwork(object):
    def __init__(self, input_row, input_col, params):
        super(CnnNetwork, self).__init__()
        self.input_row = input_row
        self.input_col = input_col
        self.params = params
        self.network = None

        self.network = self.conv1d()

    def conv1d(self):
        conv1d = self.params['conv1d']
        dense = self.params['dense']
        first_conv = list(chain.from_iterable(conv1d[0].values()))
        if self.input_col:
            inputs = tf.keras.Input(shape=(self.input_row, self.input_col))
            output = conv_factory(inputs, first_conv[0], first_conv[1], first_conv[2], first_conv[3],
                                  first_conv[4])
        else:
            inputs = tf.keras.Input(shape=self.input_row)
            output = Embedding(input_dim=22, output_dim=5, input_length=self.input_row)(inputs)
            output = conv_factory(output, first_conv[0], first_conv[1], first_conv[2], first_conv[3],
                                  first_conv[4])

        for value in conv1d[1:]:
            sub_conv = list(chain.from_iterable(value.values()))
            output = conv_factory(output, sub_conv[0], sub_conv[1], sub_conv[2], sub_conv[3], sub_conv[4])

        output = Flatten()(output)
        for value in dense:
            sub_dense = list(chain.from_iterable(value.values()))
            output = dense_factory(output, sub_dense[0], sub_dense[1])

        output = Dense(units=1, activation='sigmoid')(output)

        deep_model = Model(inputs=[inputs], outputs=[output], name='CnnModel')
        deep_model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return deep_model

    def get_network(self):
        return self.network


class RnnNetwork(object):
    def __init__(self, input_row, input_col, params, model_type):
        super(RnnNetwork, self).__init__()
        self.input_row = input_row
        self.input_col = input_col
        self.params = params
        self.model_type = model_type
        self.network = None

        self.network = self.rnn()

    def rnn(self):
        is_cycle = self.params['bidirectional']
        rnn = self.params['rnn']
        dense = self.params['dense']
        first_rnn = list(chain.from_iterable(rnn[0].values()))
        if self.input_col:
            inputs = tf.keras.Input(shape=(self.input_row, self.input_col))
            output = rnn_factory(inputs, self.model_type, first_rnn, is_cycle)
        else:
            inputs = tf.keras.Input(shape=self.input_row)
            output = Embedding(input_dim=22, output_dim=5, input_length=self.input_row)(inputs)
            output = rnn_factory(output, self.model_type, first_rnn, is_cycle)

        for value in rnn[1:]:
            sub_rnn = list(chain.from_iterable(value.values()))
            output = rnn_factory(output, self.model_type, sub_rnn, is_cycle)

        output = Flatten()(output)
        for value in dense:
            sub_dense = list(chain.from_iterable(value.values()))
            output = dense_factory(output, sub_dense[0], sub_dense[1])

        output = Dense(units=1, activation='sigmoid')(output)

        deep_model = Model(inputs=[inputs], outputs=[output], name='RnnModel')
        deep_model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return deep_model

    def get_network(self):
        return self.network
