import os.path
import re
import time
import argparse
import pandas as pd
from Bio import SeqIO
from Scripts import util_encode
import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


def file_check(filename):
    if not os.path.exists(filename):
        print("file does not exist")
        sys.exit()
    else:
        with open(filename, "r") as handle:
            fasta = SeqIO.parse(handle, "fasta")
            return any(fasta)


def get_dataset(filepath):
    try:
        predict_id = []
        predict_seq = []
        for index, record in enumerate(SeqIO.parse(filepath, 'fasta')):
            re_search = re.search(r"\|[-A-Za-z0-9]+\|", record.name)
            if re_search:
                name = re_search.group()[1:-1]
            else:
                name = record.name
            sequences = 'X' * 19 + str(record.seq) + 'X' * 19
            for location, seq in enumerate(sequences):
                if seq == 'K':
                    predict_id.append(name + '*' + str(location + 1 - 19))
                    predict_seq.append(sequences[location - 19:location + 20])
        csvfile = pd.DataFrame({'Protein': predict_id, 'Sequence': predict_seq})

        return csvfile
    except:
        return pd.DataFrame()


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

    return multiply([input_feature, cbam_feature])


def res_net_block(input_data, filters, strides=1):
    x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if strides != 1:
        down_sample = layers.Conv1D(filters, kernel_size=1, strides=strides)(input_data)
    else:  # 否就直接连接
        down_sample = input_data
    x = layers.Add()([x, down_sample])
    output = layers.Activation('relu')(x)
    return output


def HDeepSPred(Encode):
    inputs = tf.keras.Input(shape=(Encode.shape[1], Encode.shape[2]))
    x = layers.Conv1D(128, kernel_size=1)(inputs)
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

    z = layers.Bidirectional(layers.GRU(32, return_sequences=True, dropout=0.3, recurrent_initializer='orthogonal'))(
        inputs)
    z = layers.Bidirectional(layers.GRU(32, return_sequences=True, dropout=0.3, recurrent_initializer='orthogonal'))(z)

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


def predict(dataframe, model_path, save_path):
    sequences = list(dataframe['Sequence'])
    x = util_encode.onehot(sequences)
    sign = list(dataframe['Protein'])
    name = []
    position = []
    for s in sign:
        reversal = s[::-1]
        site = reversal.index('*')
        name.append(reversal[site + 1:][::-1])
        position.append(reversal[:site][::-1])
    folds = [0, 1, 2, 3, 4]
    predict_score = np.zeros((len(x), len(folds)))
    predict_result = []
    predict_confidence = []
    for fold in folds:
        modelName = '_' + str(fold) + '.h5'
        modelPath = os.path.join(model_path, modelName)
        network = HDeepSPred(x)
        network.load_weights(modelPath)
        predict_score[:, fold - 1:fold] = network.predict(x)

    predict_average_score = np.average(predict_score, axis=1)
    predict_average_score = np.around(predict_average_score, 3)
    for i in predict_average_score:
        if i >= 0.85:
            predict_result.append(1)
            predict_confidence.append('Very High confidence')
        elif i >= 0.7:
            predict_result.append(2)
            predict_confidence.append('High confidence')
        elif i >= 0.5:
            predict_result.append(3)
            predict_confidence.append('Medium confidence')
        else:
            predict_result.append(0)
            predict_confidence.append('No')
    saveCsv = pd.DataFrame({'Protein': name, 'Position': position, 'Sequence': dataframe['Sequence'],
                            'Prediction score': predict_average_score, 'Prediction category': predict_confidence})
    saveCsv.to_csv(save_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",
                                     description="ResSUMO: A Deep Learning Architecture Based on Residual Structure "
                                                 "for Lysine SUMOylation Sites Prediction")
    parser.add_argument("--file", required=True, help="input fasta format file")

    parent_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    args = parser.parse_args()
    filecheck = file_check(args.file)
    if filecheck:
        dataset = get_dataset(args.file)
        net_path = os.path.join(os.getcwd(), 'Models')
        res_path = os.path.join(os.getcwd(), 'Result')
        result_path = os.path.join(res_path, str(time.time()).split('.')[0] + '.csv')
        predict(dataset, net_path, result_path)
    else:
        print("The input file format is incorrect, it must be in fasta format")
        sys.exit()
    print("The prediction results are stored in ", result_path)
