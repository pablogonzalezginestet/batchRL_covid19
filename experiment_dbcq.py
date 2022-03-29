'''
This module defines the Experiment class that intializes, trains, and evaluates a Recurrent autoencoder.

The central focus of this class is to develop representations of sequential patient states in acute clinical settings.
These representations are learned through an auxiliary task of predicting the subsequent physiological observation but 
are also used to train a treatment policy via offline RL. The specific policy learning algorithm implemented through this
module is the discretized form of Batch Constrained Q-learning [Fujimoto, et al (2019)]

This module was designed and tested for use with a Septic patient cohort extracted from the MIMIC-III (v1.4) database. It is
assumed that the data used to create the Dataloaders in lines 174, 180 and 186 is patient and time aligned separate sequences 
of:
    (1) patient demographics
    (2) observations of patient vitals, labs and other relevant tests
    (3) assigned treatments or interventions
    (4) how long each patient trajectory is
    (5) corresponding patient acuity scores, and
    (6) patient outcomes (here, binary - death vs. survival)

The cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:
 - The code for the AIS approach and general framework we build from was developed by Jayakumar Subramanian

'''
import numpy as np
import pandas as pd
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from itertools import chain

#import signatory

from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from utils import one_hot, ReplayBuffer
import os
import copy
import pickle

from dBCQ_utils import *

#from models import AE, AIS, CDE, DST, DDM, RNN, ODERNN
from models import RNN
from models.common import get_dynamics_losses, pearson_correlation, mask_from_lengths

class Experiment(object): 
    def __init__(self, domain, train_data_file, test_data_file, minibatch_size, rng, device,
                 behav_policy_file_wDemo, behav_policy_file,
                context_input=False, context_dim=8, drop_smaller_than_minibatch=True, 
                folder_name='/Name', autoencoder_saving_period=20, resume=False, sided_Q='negative',  
                autoencoder_num_epochs=50, autoencoder_lr=0.001, autoencoder='RNN', hidden_size=16, ais_gen_model=1, 
                ais_pred_model=1, embedding_dim=4, state_dim=2, num_actions=5, corr_coeff_param=10, dst_hypers = {},
                 cde_hypers = {}, odernn_hypers = {},  **kwargs):
        '''
        We assume discrete actions and scalar rewards!
        '''

        self.rng = rng
        self.device = device
        self.train_data_file = train_data_file
        #self.validation_data_file = validation_data_file
        self.test_data_file = test_data_file
        self.minibatch_size = minibatch_size
        self.drop_smaller_than_minibatch = drop_smaller_than_minibatch
        self.autoencoder_num_epochs = autoencoder_num_epochs 
        self.autoencoder = autoencoder
        self.autoencoder_lr = autoencoder_lr
        self.saving_period = autoencoder_saving_period
        self.resume = resume
        self.sided_Q = sided_Q
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.corr_coeff_param = corr_coeff_param

        self.context_input = context_input # Check to see if we'll one-hot encode the categorical contextual input
        self.context_dim = context_dim # Check to see if we'll remove the context from the input and only use it for decoding
        self.hidden_size = hidden_size
        
        if self.context_input:
            self.input_dim = self.state_dim + self.context_dim + self.num_actions
        else:
            self.input_dim = self.state_dim + self.num_actions
        
        self.autoencoder_lower = self.autoencoder.lower()
        self.data_folder = folder_name + f'/{self.autoencoder_lower}_data'
        self.checkpoint_file = folder_name + f'/{self.autoencoder_lower}_checkpoints/checkpoint.pt'
        if not os.path.exists(folder_name + f'/{self.autoencoder_lower}_checkpoints'):
            os.mkdir(folder_name + f'/{self.autoencoder_lower}_checkpoints')
        if not os.path.exists(folder_name + f'/{self.autoencoder_lower}_data'):
            os.mkdir(folder_name + f'/{self.autoencoder_lower}_data')
        self.store_path = folder_name
        #self.gen_file = folder_name + f'/{self.autoencoder_lower}_data/{self.autoencoder_lower}_gen.pt'
        #self.pred_file = folder_name + f'/{self.autoencoder_lower}_data/{self.autoencoder_lower}_pred.pt'
        
        
        #elif self.autoencoder == 'RNN':
        #self.container = RNN.ModelContainer(device)
        
        #self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.num_actions, context_input=self.context_input, context_dim=self.context_dim)
        #self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.num_actions)
            
        #else:
        #    raise NotImplementedError

        self.buffer_save_file = self.data_folder + '/ReplayBuffer_data'
        self.next_obs_pred_errors_file = self.data_folder + '/test_next_obs_pred_errors.pt'
        #self.test_representations_file = self.data_folder + '/test_representations.pt'
        #self.test_correlations_file = self.data_folder + '/test_correlations.pt'
        self.policy_eval_save_file = self.data_folder + '/dBCQ_policy_eval_raw_data'
        self.policy_save_file = self.data_folder + '/dBCQ_policy_raw_data'
        self.behav_policy_file_wDemo = behav_policy_file_wDemo
        self.behav_policy_file = behav_policy_file
        
        
        # Read in the data csv files
        assert (domain=='sepsis')        
        self.train_demog, self.train_states, self.train_interventions, self.train_lengths, self.train_times, self.rewards, self.dones = torch.load(self.train_data_file)
        train_idx = torch.arange(self.train_demog.shape[0])
        self.train_dataset = TensorDataset(self.train_demog, self.train_states, self.train_interventions,self.train_lengths,self.train_times, self.rewards, train_idx, self.dones)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.minibatch_size, shuffle=True)

        #self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards = torch.load(self.validation_data_file)
        #val_idx = torch.arange(self.val_demog.shape[0])
        #self.val_dataset = TensorDataset(self.val_demog, self.val_states, self.val_interventions, self.val_lengths, self.val_times, self.val_acuities, self.val_rewards, val_idx)

        #self.val_loader = DataLoader(self.val_dataset, batch_size=self.minibatch_size, shuffle=False)

        self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_rewards, self.test_dones = torch.load(self.test_data_file)
        test_idx = torch.arange(self.test_demog.shape[0])
        self.test_dataset = TensorDataset(self.test_demog, self.test_states, self.test_interventions, self.test_lengths, self.test_times, self.test_rewards, test_idx, self.test_dones)

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.minibatch_size, shuffle=False)
        
                  
            
    
    #def load_model_from_checkpoint(self, checkpoint_file_path):
    #    checkpoint = torch.load(checkpoint_file_path)
    #    self.gen.load_state_dict(checkpoint['{}_gen_state_dict'.format(self.autoencoder.lower())])
    #    self.pred.load_state_dict(checkpoint['{}_pred_state_dict'.format(self.autoencoder.lower())])
        #if self.autoencoder == 'DDM':
        #    self.dyn.load_state_dict(checkpoint['{}_dyn_state_dict'.format(self.autoencoder.lower())])
    #    print("Experiment: generator and predictor models loaded.")
        
    def evaluate_trained_model(self):
        '''After training, this method can be called to use the trained autoencoder to embed all the data in the representation space.
        We encode all data subsets (train, validation and test) separately and save them off as independent tuples. We then will
        also combine these subsets to populate a replay buffer to train a policy from.
        
        This method will also evaluate the decoder's ability to correctly predict the next observation from the and also will
        evaluate the trained representation's correlation with the acuity scores.
        '''

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer(self.state_dim + self.context_dim, self.minibatch_size, 350000, self.device, encoded_state=False, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))

        errors = []
        #correlations = torch.Tensor()
        #test_representations = torch.Tensor()
        print('Encoding the Training and Validataion Data.')
        ## LOOP THROUGH THE DATA
        # -----------------------------------------------
        # For Training and Validation sets (Encode the observations only, add all data to the experience replay buffer)
        # For the Test set:
        # - Encode the observations
        # - Save off the data (as test tuples and place in the experience replay buffer)
        # - Evaluate accuracy of predicting the next observation using the decoder module of the model
        # - Evaluate the correlation coefficient between the learned representations and the acuity scores
        with torch.no_grad():
            for i_set, loader in enumerate([self.train_loader,  self.test_loader]):
            #for i_set, loader in enumerate([self.train_loader, self.val_loader, self.test_loader]):
                if i_set == 1:
                    print('Encoding the Test Data. Evaluating prediction accuracy. Calculating Correlation Coefficients.')
                for dem, ob, ac, l, t, rewards, idx, dones in loader:
                    dem = dem.to(self.device)
                    ob = ob.to(self.device)
                    ac = ac.to(self.device)
                    l = l.to(self.device)
                    t = t.to(self.device)
                    #scores = scores.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    max_length = int(l.max().item())

                    ob = ob[:,:max_length,:]
                    dem = dem[:,:max_length,:]
                    ac = ac[:,:max_length,:]
                    #scores = scores[:,:max_length,:]
                    rewards = rewards[:,:max_length]
                    dones = dones[:,:max_length]

                    cur_obs, next_obs = ob[:,:-1,:], ob[:,1:,:]
                    cur_dem, next_dem = dem[:,:-1,:], dem[:,1:,:]
                    cur_actions = ac[:,:-1,:]
                    cur_rewards = rewards[:,:-1]
                    done = dones[:,1:]
                    #cur_scores = scores[:,:-1,:]
                    mask = (cur_obs==0).all(dim=2)
                                        
                    cur_actions = cur_actions[~mask].cpu()
                    cur_rewards = cur_rewards[~mask].cpu()
                    cur_obs = cur_obs[~mask].cpu()  # Need to keep track of the actual observations that were made to form the corresponding representations (for downstream WIS)
                    next_obs = next_obs[~mask].cpu()
                    cur_dem = cur_dem[~mask].cpu()
                    next_dem = next_dem[~mask].cpu()
                    done = done[~mask].cpu()
                    
                    # Loop over all transitions and add them to the replay buffer
                    for i_trans in range(cur_obs.shape[0]):
                        #done = cur_rewards[i_trans] != 0
                        self.replay_buffer.add(torch.cat((cur_obs[i_trans],cur_dem[i_trans]),dim=-1).numpy(),cur_actions[i_trans].argmax().item(), torch.cat((next_obs[i_trans], next_dem[i_trans]), dim=-1).numpy(), cur_rewards[i_trans].item(), done[i_trans].item())
                        
            ## SAVE OFF DATA
            # --------------
            self.replay_buffer.save(self.buffer_save_file)
            #torch.save(errors, self.next_obs_pred_errors_file)
            #torch.save(test_representations, self.test_representations_file)
            #torch.save(correlations, self.test_correlations_file)
  
  
    def train_dBCQ_policy(self, pol_learning_rate=1e-3):

        # Initialize parameters for policy learning
        params = {
            "eval_freq": 500,
            "discount": 0.99,
            "buffer_size": 350000,
            "batch_size": self.minibatch_size,
            "optimizer": "Adam",
            "optimizer_parameters": {
                "lr": pol_learning_rate
            },
            "train_freq": 1,
            "polyak_target_update": True,
            "target_update_freq": 1,
            "tau": 0.01,
            "max_timesteps": 5e5,
            "BCQ_threshold": 0.3,
            "buffer_dir": self.buffer_save_file,
            "policy_file": self.policy_save_file+f'_l{pol_learning_rate}.pt',
            "pol_eval_file": self.policy_eval_save_file+f'_l{pol_learning_rate}.npy',
        }
        
        # Initialize a dataloader for policy evaluation (will need representations, observations, demographics, rewards and actions from the test dataset)       
        test_demog, test_states, test_interventions, test_lengths, test_times, test_rewards, test_dones = torch.load(self.test_data_file )    #'./data/rewards_new_cases/gap_5/test_set_tuples')
        #test_demog, test_states, test_interventions, test_lengths, test_times, test_rewards = torch.load('./data/test_set_tuples')
        # Initialize and Load the experience replay buffer corresponding with the current settings of rand_num, hidden_size, etc...
        replay_buffer = ReplayBuffer(self.state_dim + self.context_dim, self.minibatch_size, 350000, self.device, encoded_state=False, obs_state_dim=self.state_dim + (self.context_dim if self.context_input else 0))
        # Load the pretrained policy for whether or not the demographic context was used to train the representations 
        behav_input = self.state_dim + (self.context_dim if self.context_input else 0)
        #behav_input = self.hidden_size
        behav_pol = FC_BC(behav_input, self.num_actions, 128).to(self.device)
        if self.context_input:
            behav_pol.load_state_dict(torch.load(self.behav_policy_file_wDemo))
        else:
            behav_pol.load_state_dict(torch.load(self.behav_policy_file))
        behav_pol.eval()

        # Run dBCQ_utils.train_dBCQ
        train_dBCQ1(replay_buffer, self.num_actions, self.state_dim + self.context_dim, self.device, params, behav_pol, self.context_input,test_demog, test_states, test_interventions, test_lengths, test_times, test_rewards,test_dones) # this is trained using the representation

    ####################################
    