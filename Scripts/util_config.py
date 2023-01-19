import os
import yaml
from Scripts import Nets


def get_yaml():
    now_path = os.getcwd()
    yaml_path = os.path.join(now_path, 'config.yaml')
    with open(yaml_path, encoding='utf-8') as yaml_file:
        config = yaml.safe_load(yaml_file)
    yaml_file.close()
    return config


def get_hyper_parameters():
    return get_yaml()['Net']


def get_network(row, col):
    params = get_yaml()
    Net = params['Net']
    default_model = params['Default model']
    if default_model['is_available']:
        select = default_model['types']
        if select == 0:
            network = Nets.CNN(row, col)
        elif select == 1:
            network = Nets.RSCNN(row, col)
        elif select == 2:
            network = Nets.BiLSTM(row, col)
        elif select == 3:
            network = Nets.BiGRU(row, col)
        elif select == 4:
            network = Nets.HDeepSPred(row, col)
        else:
            network = Nets.HDeepSPred(row, col)
        return network
    else:
        net_type = Net['types']
        if net_type == 0:
            model_params = params['Cnn']
            network = Nets.CnnNetwork(row, col, model_params)
        elif net_type == 1:
            model_params = params['Rnn']
            network = Nets.RnnNetwork(row, col, model_params, net_type)
        elif net_type == 2:
            model_params = params['Rnn']
            network = Nets.RnnNetwork(row, col, model_params, net_type)
        else:
            model_params = params['Cnn']
            network = Nets.CnnNetwork(row, col, model_params)
        return network
