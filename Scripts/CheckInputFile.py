import argparse
import os
import re
import random


class Sequence(object):
    def __init__(self, file):
        self.file = file  # whole file path
        self.fasta_list = []  # 2-D list [sampleName, sequence, label, training or testing]
        self.sequence_type = ''  # DNA, RNA or Protein
        self.is_equal = False  # bool: sequence with equal length?
        self.length = 0  # int
        self.error_msg = ''  # string
        self.sequence_number = 0  # int: the number of samples

        self.fasta_list, self.error_msg = self.read_fasta()
        self.sequence_number = len(self.fasta_list)

        if self.sequence_number > 0:
            self.is_equal, self.length = self.sequence_with_equal_length()
            self.sequence_type = self.check_sequence_type()
        else:
            if self.error_msg == '':
                self.error_msg = 'File format error.'

    def read_fasta(self):
        """
        read fasta sequence
        :return:
        """
        msg = ''
        if not os.path.exists(self.file):
            msg = 'Error: file %s does not exist.' % self.file
            return [], msg
        if not os.path.isfile(self.file):
            msg = 'Error: file %s does not a file.' % self.file
            return [], msg
        with open(self.file) as f:
            records = f.read()
        f.close()
        records = records.split('>')[1:]
        fasta_sequences = []
        for fasta in records:
            array = fasta.split('\n')
            header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTUVWYX]', 'X', ''.join(array[1:]).upper())
            header_array = header.split('|')
            name = header_array[0]
            label = header_array[1] if len(header_array) >= 2 else '0'
            label_train = header_array[2] if len(header_array) >= 3 else 'training'
            fasta_sequences.append([name, sequence, label, label_train])
        if len(fasta_sequences) == 0:
            msg = 'Error: file %s content error.' % self.file
        return fasta_sequences, msg

    def sequence_with_equal_length(self):
        """
        Check if fasta sequence is in equal length
        :return:
        """
        length_set = set()
        for item in self.fasta_list:
            length_set.add(len(item[1]))
        length_set = sorted(length_set)
        if len(length_set) == 1:
            return True, length_set[0]
        else:
            return False, length_set[0]

    def check_sequence_type(self):
        """
        Specify sequence type (Protein, DNA or RNA)
        :return:
        """
        tmp_fasta_list = []
        if len(self.fasta_list) < 100:
            tmp_fasta_list = self.fasta_list
        else:
            random_index = random.sample(range(0, len(self.fasta_list)), 100)
            for i in random_index:
                tmp_fasta_list.append(self.fasta_list[i])

        sequence = ''
        for item in tmp_fasta_list:
            sequence += item[1]
        char_set = set(sequence)
        if 5 < len(char_set) <= 21:
            for line in self.fasta_list:
                line[1] = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', 'X', line[1])
            return 'Protein'
        elif 0 < len(char_set) <= 5 and 'T' in char_set:
            for line in self.fasta_list:
                line[1] = re.sub('T', 'U', line[1])
            return 'RNA'
        elif 0 < len(char_set) <= 5 and 'U' in char_set:
            for line in self.fasta_list:
                line[1] = re.sub('U', 'T', line[1])
            return 'DNA'
        else:
            return 'Unknown'

    def print_result(self):

        length = "yes, length is " + str(self.length) if self.is_equal else "no."
        print("==================================================")
        if self.error_msg:
            print(self.error_msg)
        else:
            print("file name: %s" % os.path.basename(self.file))
            print("sequence type: %s" % self.sequence_type)
            print("sequence with equal length: %s" % length)
            print("the number of samples is: %d" % self.sequence_number)
            print("==================================================")

    def get_fasta_list(self):
        return self.fasta_list

    def get_file(self):
        return self.file

    def get_error_msg(self):
        return self.error_msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",
                                     description="Check that the format and content of the input file are correct")
    parser.add_argument("--file", required=True, help="input fasta formate file")

    args = parser.parse_args()

    seq = Sequence(args.file)

    seq.print_result()
