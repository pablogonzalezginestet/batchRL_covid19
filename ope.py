

import sys
import pickle
#import click
import yaml


import torch
import numpy as np
import pandas as pd
import datetime as dt
import random
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import os
import glob
import copy

from sklearn.model_selection import train_test_split, KFold
import operator



def ReplayBuffer_test(df, state_features):
    #
    df_temp = df.copy()
    df_temp = df_temp.reset_index()
    a = df_temp.copy()
    batch_size = len(a)
    #
    input_size = len(state_features)
    states = np.zeros((batch_size,input_size))
    actions = np.zeros((batch_size,1))
    rewards = np.zeros((batch_size,1))
    next_states = np.zeros((batch_size,input_size))
    done_flags =  np.zeros((batch_size,1))
    country = []    
    counter = 0
    for idx,obser in a.iterrows():
        index = int(obser['index'])
        state = obser[state_features]
        action = int(obser['action'])
        reward = obser['reward_5']
        #
        if int(obser['done']) == 1:
            next_state = np.zeros(len(state))
        else:
            next_state = df_temp[df_temp['index']== index+1][state_features] # here I go to look for the whole dataset the next state for that country, its trajectory
        #
        states[counter,:] = state
        next_states[counter,:] = next_state 
        actions[counter] = action
        rewards[counter] = reward
        done_flags[counter] = int(obser['done'])
        country.append( obser['Entity'])
        counter +=1
    #        
    return torch.Tensor(np.array(states,dtype=np.float)), torch.Tensor(actions) , torch.Tensor(np.array(rewards,dtype=np.float)), torch.Tensor(np.array(next_states,dtype=np.float)), torch.Tensor(done_flags), country

# Simple full-connected supervised network for Behavior Cloning of batch data
class FC_BC(nn.Module):
    def __init__(self, state_dim=10, num_actions=5, num_nodes=64):
        super(FC_BC, self).__init__()
        self.l1 = nn.Linear(state_dim, num_nodes)
        self.bn1 = nn.BatchNorm1d(num_nodes)
        self.l2 = nn.Linear(num_nodes, num_nodes)
        self.bn2 = nn.BatchNorm1d(num_nodes)
        self.l3 = nn.Linear(num_nodes, num_actions)
    def forward(self, state):
        out = F.relu(self.l1(state))
        out = self.bn1(out)
        out = F.relu(self.l2(out))
        out = self.bn2(out)
        return self.l3(out)

        
        
# Simple fully-connected Q-network for the policy in DQN
class FC_Q_dqn(nn.Module):
    def __init__(self, state_dim, num_actions, num_nodes=128):
        super(FC_Q_dqn, self).__init__()
        self.q1 = nn.Linear(state_dim, num_nodes)
        self.q2 = nn.Linear(num_nodes, num_nodes)
        self.q3 = nn.Linear(num_nodes, num_actions)
    #
    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))
        return self.q3(q)


        
state_features = [  'stringency_index', 'new_cases',
                    'population_density', 'gdp_per_capita', 'diabetes_prevalence',
                    'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
                    'life_expectancy', 'hospital_beds_per_thousand' ]
state_dim = len(state_features)
num_actions= 5   
device='cuda'
behav_policy_file_wDemo= './behavcloning/test/new_cases_gap5/BC_model.pt'
df = pd.read_csv('C:/Users/pabgon/rl_representations/data/df_rewards_reproduction_rate_23March2022.csv', dtype='object', header=0)
df_normalized = copy.deepcopy(df)
temp=df_normalized[state_features].astype(float).describe()
means=temp.loc['mean'].values
stds=temp.loc['std'].values
df_normalized[state_features] = (df_normalized[state_features].astype(float)-means)/stds
df_normalized['reward'] = pd.to_numeric(df_normalized['reward_5'])
### DQN ###
behav_pol = FC_BC(state_dim, num_actions, 128)
behav_pol.load_state_dict(torch.load(behav_policy_file_wDemo)) 

# 1 
Q = FC_Q_dqn(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/DQN/DQN_policy_data_rnd_state1_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/DQN/countries_test_set_rnd_state1.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
#


V_WIS_1 = []
V_gov_1 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_1.append(np.clip(reward_disc,-500,500))
    V_gov_1.append(reward_disc_gov)

np.nanmean(V_WIS_1)
# 2 
Q = FC_Q_dqn(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/DQN/DQN_policy_data_rnd_state2_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/DQN/countries_test_set_rnd_state2.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
    
V_WIS_2 = []
V_gov_2 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_2.append(np.clip(reward_disc,-500,500))
    V_gov_2.append(reward_disc_gov)    

# 3 
Q = FC_Q_dqn(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/DQN/DQN_policy_data_rnd_state3_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/DQN/countries_test_set_rnd_state3.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
    
V_WIS_3 = []
V_gov_3 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_3.append(np.clip(reward_disc,-500,500)) 
    V_gov_3.append(reward_disc_gov)    

# 4 
Q = FC_Q_dqn(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/DQN/DQN_policy_data_rnd_state4_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/DQN/countries_test_set_rnd_state4.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
    
V_WIS_4 = []
V_gov_4 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_4.append(np.clip(reward_disc,-500,500)) 
    V_gov_4.append(reward_disc_gov)
    
# 5 
Q = FC_Q_dqn(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/DQN/DQN_policy_data_rnd_state5_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/DQN/countries_test_set_rnd_state5.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)

    
V_WIS_5 = []
V_gov_5 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state), dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
        reward_disc_gov += (0.99**t) * np.array(reward[t])
    V_WIS_5.append(np.clip(reward_disc,-500,500))
    V_gov_5.append(reward_disc_gov)


np.nanmean([V_WIS_1,V_WIS_2,V_WIS_3,V_WIS_4,V_WIS_5])
np.nanmean([V_gov_1,V_gov_2,V_gov_3,V_gov_4,V_gov_5])

# DBCQ

class FC_Q(nn.Module):
    def __init__(self, state_dim, num_actions, num_nodes=128):
        super(FC_Q, self).__init__()
        self.q1 = nn.Linear(state_dim, num_nodes)
        self.q2 = nn.Linear(num_nodes, num_nodes)
        self.q3 = nn.Linear(num_nodes, num_actions)
        self.i1 = nn.Linear(state_dim, num_nodes)
        self.i2 = nn.Linear(num_nodes, num_nodes)
        self.i3 = nn.Linear(num_nodes, num_actions)
    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))
        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        i = F.relu(self.i3(i))
        return self.q3(q), F.log_softmax(i, dim=1), i
        
# 1        
Q = FC_Q(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_data_rnd_state1_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/dBCQ/countries_test_set_rnd_state1.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
#


V_WIS_dbcq_1 = []
#V_gov_1 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_dbcq_1.append(np.clip(reward_disc,-500,500))
    #V_gov_1.append(reward_disc_gov)

np.nanmean(V_WIS_dbcq_1)

# 2
Q = FC_Q(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_data_rnd_state2_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/dBCQ/countries_test_set_rnd_state2.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
#


V_WIS_dbcq_2 = []
#V_gov_1 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_dbcq_2.append(np.clip(reward_disc,-500,500))
    #V_gov_1.append(reward_disc_gov)

np.nanmean(V_WIS_dbcq_2)

# 3 
Q = FC_Q(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_data_rnd_state3_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/dBCQ/countries_test_set_rnd_state3.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
#


V_WIS_dbcq_3 = []
#V_gov_1 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_dbcq_3.append(np.clip(reward_disc,-500,500))
    #V_gov_1.append(reward_disc_gov)

np.nanmean(V_WIS_dbcq_3)

# 4
Q = FC_Q(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_data_rnd_state4_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/dBCQ/countries_test_set_rnd_state4.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
#


V_WIS_dbcq_4 = []
#V_gov_1 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_dbcq_4.append(np.clip(reward_disc,-500,500))
    #V_gov_1.append(reward_disc_gov)

np.nanmean(V_WIS_dbcq_4)

# 5
Q = FC_Q(state_dim, num_actions)
Q.load_state_dict(torch.load('./srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_data_rnd_state5_lr0.001.pt')['policy_Q_function']) 
countries = np.load('./srl_test/results/rewards_gap5/states_t/dBCQ/countries_test_set_rnd_state5.npy',allow_pickle = True)

rho_o_t = np.zeros((len(countries),500))
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    for t in range(len_traj):
        rho_o_t[j,t] = np.clip(ratio_prob[0:t].prod(),1e-30, 1e4)
#


V_WIS_dbcq_5 = []
#V_gov_1 = []
for j in range(len(countries)):
    idx = countries[j]
    df_temp = df_normalized[df_normalized['Entity'] == idx ]
    state = torch.Tensor(np.array(df_temp[state_features],dtype=np.float))
    df_temp['action'] = pd.to_numeric(df_temp['facial_coverings']) 
    action = torch.tensor(np.array(df_temp['action']),dtype=torch.long)
    reward =  torch.Tensor(np.array(df_temp['reward_5'],dtype=np.float))
    p_obs_traj = (F.softmax(behav_pol(state),dim=-1)).gather(1,action.unsqueeze(1)) #all this is for the obs action
    p_new_traj = (F.softmax(Q(state)[0], dim=-1)).gather(1,action.unsqueeze(1))
    ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
    len_traj = reward.shape[0]
    reward_disc = 0
    reward_disc_gov = 0
    for t in range(len_traj):
        reward_disc_gov += (0.99**t) * np.array(reward[t])
        reward_disc += (0.99**t) * np.array(reward[t]) * ( ratio_prob[0:t+1].prod()/ (rho_o_t[:,t].sum()/len(countries)) )
    V_WIS_dbcq_5.append(np.clip(reward_disc,-500,500))
    #V_gov_1.append(reward_disc_gov)

np.nanmean(V_WIS_dbcq_5)

#########################################

np.nanmean([V_WIS_1,V_WIS_2,V_WIS_3,V_WIS_4,V_WIS_5])
np.nanmean([V_gov_1,V_gov_2,V_gov_3,V_gov_4,V_gov_5])
np.nanmean([V_WIS_dbcq_1,V_WIS_dbcq_2,V_WIS_dbcq_3,V_WIS_dbcq_4,V_WIS_dbcq_5])
