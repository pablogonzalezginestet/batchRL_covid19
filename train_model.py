

import random
import os
import sys
import pickle
#import click
import yaml
import numpy as np
from experiment import Experiment

import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
np.set_printoptions(suppress=True, linewidth=200, precision=2)

#@click.command()
#@click.option('--autoencoder', '-a', type=click.Choice(['AE', 'AIS', 'CDE', 'DDM', 'DST', 'ODERNN', 'RNN']))
#@click.option('--domain', '-d', default='sepsis', help="Only 'sepsis' implemented for now")
#@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(autoencoder, domain):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = yaml.safe_load(open(os.path.join(dir_path, '../configs/common.yaml'), 'r'))    
    cfg_file = os.path.join(dir_path, '../configs/config_' + domain + '_rnn.yaml')
    model_params = yaml.safe_load(open(cfg_file, 'r'))
    
    #if autoencoder == 'CDE':
    #    model_params['coefs_folder'] =  os.path.join(params['storage_path'], model_params['coefs_folder'])
            
    for i in model_params:
        params[i] = model_params[i]        

    print('Parameters')
    for key in params:
        print(key, params[key])
    print('=' * 30)

    # process param keys and values to match input to Cortex
    params['device'] = torch.device(params["device"])
    random_seed = params['random_seed']
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random_state = np.random.RandomState(random_seed)
    params['rng'] = random_state
    params['domain'] = domain
        
    folder_name = params['storage_path'] + params['folder_location'] + params['folder_name']
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    params['folder_name'] = folder_name
    
    torch.set_num_threads(torch.get_num_threads())
    
    params[f'{autoencoder.lower()}_hypers'] = model_params # Cortex hyperparameter dictionaries 
    
    # Experiment
    experiment = Experiment(**params)    
    experiment.train_autoencoder()
    experiment.evaluate_trained_model() # this populate the BufferReplay
    #experiment.run_behavcloning() 
    experiment.train_dBCQ_policy(params['pol_learning_rate']) 
    print('=' * 30)
    
    # i think that i will do OPE computing for each trajectory the recursion formula 

    #with open(folder_name + '/config.yaml', 'w') as y:
    #    yaml.safe_dump(params, y)  # saving params for reference

if __name__ == '__main__':
    run('RNN','sepsis')