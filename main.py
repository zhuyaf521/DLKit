import argparse
import sys
from datetime import datetime
import tensorflow as tf
from Scripts import CheckInputFile, util_encode, util_funs, util_config
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np


def evaluate(data_list, out, code_type, seq_type):
    flag = True
    sequence, y_train, y_test = util_encode.get_row_data(data_list)
    test = sequence[1] if sequence[1] else None
    x_train, x_test = util_encode.code(sequence[0], test, code_type, seq_type)

    classes = sorted(list(set(y_test)))
    prediction_result_cv = []
    prediction_result_ind = []
    params = util_config.get_hyper_parameters()
    folds = StratifiedKFold(params['cross_validation']).split(x_train, y_train)

    for i, (trained, validated) in enumerate(folds):
        X_train, Y = x_train[trained], y_train[trained]
        X_test, y = x_train[validated], y_train[validated]

        if len(x_train.shape) == 3:
            network = util_config.get_network(x_train.shape[1], x_train.shape[2]).get_network()
        else:
            network = util_config.get_network(x_train.shape[1], None).get_network()
        if flag:
            network.summary()
            print("===================================================================================")
            print("sequence type: " + seq_type)
            print("===================================================================================")
            state = input("The model structure is shown above,whether to start training (Yes or No): ")
            if state.upper() == 'YES':
                util_funs.mkdir(out)
            else:
                sys.exit()
            flag = False

        if not os.path.exists(os.path.join(out, '_%d.h5' % i)):
            best_saving = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(out, '_%d.h5' % i),
                                                             monitor='val_loss',
                                                             verbose=1, save_best_only=True, save_weights_only=True)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience'])
            history = network.fit(X_train, Y, epochs=params['epochs'], validation_data=(X_test, y), verbose=2,
                                  callbacks=[best_saving, early_stopping], batch_size=params['batch_size'])
            acc_loss_subpath = os.path.join(out, 'history')
            util_funs.mkdir(acc_loss_subpath)
            acc_loss_path = os.path.join(acc_loss_subpath, 'fold' + str(i + 1) + '.png')
            util_funs.plot_loss(history, acc_loss_path)
        network.load_weights(os.path.join(out, '_%d.h5' % i))
        tmp_result = np.zeros((len(y), len(classes)))
        tmp_result[:, 0], tmp_result[:, 1] = y, network.predict(X_test)[:, 0]
        prediction_result_cv.append(tmp_result)

        if x_test is not None:
            tmp_result1 = np.zeros((len(y_test), len(classes)))
            tmp_result1[:, 0], tmp_result1[:, 1] = y_test, network.predict(x_test)[:, 0]
            prediction_result_ind.append(tmp_result1)
    if x_test is not None:
        return prediction_result_cv, prediction_result_ind
    else:
        return prediction_result_cv, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",
                                     description="Check that the format and content of the input file are correct")
    parser.add_argument("--file", required=True, help="input fasta format file")

    parser.add_argument("--type", required=False,
                        choices=['OneHot', 'WordEmbedding'], default='OneHot',
                        help="the encoding type")

    args = parser.parse_args()
    data = CheckInputFile.Sequence(args.file)
    fasta_list = data.get_fasta_list()
    sequence_type = data.sequence_type
    if data.get_error_msg():
        print(data.get_error_msg())
        sys.exit()
    ts = datetime.now()

    ts = str(ts).replace('-', '').replace(' ', '_').replace(':', '')[:15]

    path = os.path.join(os.getcwd(), 'Result')
    output = os.path.join(path, ts)

    pre_cv, pre_ind = evaluate(fasta_list, output, args.type, sequence_type)

    util_funs.save_result(pre_cv, output, pre_ind)

    print("The program running results are saved in: " + output)
