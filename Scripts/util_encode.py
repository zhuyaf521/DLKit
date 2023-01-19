import re
import numpy as np
import sys


def get_row_data(fasta_list):
    train_data = []
    y_train = []
    test_data = []
    y_test = []
    for item in fasta_list:
        if item[3] == 'training':
            train_data.append(item[1])
            y_train.append(item[2])
        else:
            test_data.append(item[1])
            y_test.append(item[2])
    if len(train_data) == 0:
        print('There is no training data in the current fasta file!')
        sys.exit()
    if len(test_data) == 0:
        return (train_data, None), np.array(y_train).astype(np.float64), None
    else:
        return (train_data, test_data), np.array(y_train).astype(np.float64), np.array(y_test).astype(np.float64)


def code(train, test, code_type, seq_type):
    if seq_type == 'Protein':
        if code_type == 'OneHot':
            x_train = protein_onehot(train)
            x_test = protein_onehot(test)
        elif code_type == 'WordEmbedding':
            x_train = protein_num(train)
            x_test = protein_num(test)
        else:
            x_train = protein_onehot(train)
            x_test = protein_onehot(test)

    elif seq_type == 'DNA':
        if code_type == 'OneHot':
            x_train = dna_onehot(train)
            x_test = dna_onehot(test)
        elif code_type == 'WordEmbedding':
            x_train = dna_num(train)
            x_test = dna_num(test)
        else:
            x_train = dna_onehot(train)
            x_test = dna_onehot(test)
    elif seq_type == 'RNA':
        if code_type == 'OneHot':
            x_train = rna_onehot(train)
            x_test = rna_onehot(test)
        elif code_type == 'WordEmbedding':
            x_train = rna_num(train)
            x_test = rna_num(test)
        else:
            x_train = rna_onehot(train)
            x_test = rna_onehot(test)
    else:
        print("Wrong sequence type, Not Protein, RNA or DNA")
        sys.exit()

    return x_train, x_test


def protein_onehot(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWYX'
    encodings = []
    for seq in sequences:
        seq = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(seq).upper())
        code = []
        for aa in seq:
            single_code = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                single_code.append(tag)
            code.append(single_code)
        encodings.append(code)

    return np.array(encodings).astype(np.float64)


def protein_num(sequences):
    AA = 'XARNDCQEGHILKMFPSTWYV'
    encodings = []
    for seq in sequences:
        seq = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(seq).upper())
        code = []
        for aa in seq:
            code.append(AA.index(aa))
        encodings.append(code)

    return np.array(encodings).astype(np.float64)


def rna_onehot(sequences):
    AA = 'ACGTX'
    encodings = []
    for seq in sequences:
        seq = re.sub('[^ACGTX]', 'X', ''.join(seq).upper())
        code = []
        for aa in seq:
            single_code = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                single_code.append(tag)
            code.append(single_code)
        encodings.append(code)

    return np.array(encodings).astype(np.float64)


def rna_num(sequences):
    AA = 'ACGTX'
    encodings = []
    for seq in sequences:
        seq = re.sub('[^ACGTX]', 'X', ''.join(seq).upper())
        code = []
        for aa in seq:
            code.append(AA.index(aa))
        encodings.append(code)

    return np.array(encodings).astype(np.float64)


def dna_onehot(sequences):
    AA = 'AUGCX'
    encodings = []
    for seq in sequences:
        seq = re.sub('[^AUGCX]', 'X', ''.join(seq).upper())
        code = []
        for aa in seq:
            single_code = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                single_code.append(tag)
            code.append(single_code)
        encodings.append(code)

    return np.array(encodings).astype(np.float64)


def dna_num(sequences):
    AA = 'AUGCX'
    encodings = []
    for seq in sequences:
        seq = re.sub('[^AUGCX]', 'X', ''.join(seq).upper())
        code = []
        for aa in seq:
            code.append(AA.index(aa))
        encodings.append(code)

    return np.array(encodings).astype(np.float64)
