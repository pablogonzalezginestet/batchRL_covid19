

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


def ReplayBuffer_lstm(batch_size,df,state_features, time_step ,train=True):
    df_temp = df.copy()
    df_temp = df_temp.reset_index()
    country = np.unique(df_temp['Entity'])
    country_batch = np.random.choice(country,batch_size)
    states_batch = list()
    next_states_batch = list()
    actions_batch = list()
    rewards_batch = list()
    done_flags_batch = list()
    for j in np.arange(len(country_batch)):
        episode = df_temp[df_temp['Entity']== country_batch[j]]
        point = np.random.randint(0,len(episode)+1-time_step)
        actions_batch.append(pd.to_numeric(episode[point:point+time_step]['action']))
        rewards_batch.append(pd.to_numeric(episode[point:point+time_step]['reward_5']))
        done_flags_batch.append(pd.to_numeric(episode[point:point+time_step]['done']))
        states_batch.append((episode[point:point+time_step][state_features]))
        if float(episode['done'].iloc[point+time_step-1])==1:
            next_state_temp = episode[(point+1):point+time_step][state_features]
            next_state_temp.loc[len(next_state_temp)] = 0
            next_states_batch.append(next_state_temp)
        else:
            next_states_batch.append((episode[(point+1):point+time_step+1][state_features]))
    return np.array(states_batch,dtype=np.float), np.array(actions_batch,dtype=np.float), np.array(rewards_batch,dtype=np.float), np.array(next_states_batch,dtype=np.float), np.array(done_flags_batch,dtype=np.float)


def ReplayBuffer(batch_size,df, state_features ,train=True):
    #
    df_temp = df.copy()
    df_temp = df_temp.reset_index()
    #
    if not train:
        a = df_temp.copy()
        batch_size = len(a)
    else: 
        a = df_temp.sample(n=batch_size)
    #
    input_size = len(state_features)
    states = np.zeros((batch_size,input_size))
    actions = np.zeros((batch_size,1))
    rewards = np.zeros((batch_size,1))
    next_states = np.zeros((batch_size,input_size))
    done_flags =  np.zeros((batch_size,1)) 
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
        counter +=1
    #        
    return torch.Tensor(np.array(states,dtype=np.float)), torch.Tensor(actions) , torch.Tensor(np.array(rewards,dtype=np.float)), torch.Tensor(np.array(next_states,dtype=np.float)), torch.Tensor(done_flags)

    


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
        


class discrete_BCQ(object):
    def __init__(
        self,
        batch_size,         
        num_actions,
        state_dim,
        device,
        BCQ_threshold=0.3,
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=1e3,
        tau=0.005
    ):
        self.batch_size = batch_size
        self.device = device
        # Determine network type
        self.Q = FC_Q(state_dim, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)
        self.discount = discount
        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        # self.eval_eps = eval_eps
        self.num_actions = num_actions
        # Threshold for "unlikely" actions
        self.threshold = BCQ_threshold
        # Number of training iterations
        self.iterations = 0
    def train(self, df,state_features,):
        # Sample replay buffer
        state, action, reward, next_state, done = ReplayBuffer(self.batch_size, df, state_features)
        # Compute the target Q value
        with torch.no_grad():
            action = torch.tensor(action,dtype=torch.long).to(self.device)
            state  = state.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
            reward = reward.to(self.device)            
            double_q = self.Q_target(next_state)[0].max(1, keepdim=True)[0]
            q, imt, i = self.Q(next_state)
            imt = imt.exp()
            imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()
            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)
            q, imt, i = self.Q_target(next_state)
            target_Q = 10*reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)
        # Get current Q estimate
        current_Q, imt, i = self.Q(state)
        current_Q = current_Q.gather(1, action)
        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt, action.reshape(-1))
        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()
        # Optimize the Q
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()
        return Q_loss.detach().item()
    #
    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
           target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    #
    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())             

#def train_DQN(df, batch_size ,num_actions, state_dim, device, parameters, behav_pol,is_demog, test_demog, test_states, test_interventions, test_lengths, test_times, test_rewards,test_dones):

def train_dBCQ(df, val_df,batch_size ,num_actions, state_dim, state_features, device, parameters, behav_pol):
    # For saving files
    loss_train_file = parameters['loss_train_file']
    pol_eval_file_wis = parameters['pol_eval_file_wis']
    pol_eval_file_dr = parameters['pol_eval_file_dr']
    pol_file = parameters['policy_file']
    action_RL_test_file = parameters['action_RL_file']
    action_iter_test_file = parameters['action_iter_test'] 
    Q_RL_test_file = parameters['Q_RL_file'] 
    Q_iter_test_file = parameters['Q_iter_test']       
    # Initialize and load policy
    policy = discrete_BCQ(
        batch_size,
        num_actions,
        state_dim,
        device,
        )
    evaluations_wis = []
    evaluations_dr = []
    Q_RL_test_set = []
    action_RL_test_set = []
    loss_stat = []
    episode_num = 0
    done = True
    training_iters = 0
    while training_iters < 40000:
        for _ in range(int(parameters["eval_freq"])):
            loss_stat.append(policy.train(df,state_features))
        #
        evaluations_wis.append(eval_policy_wis(val_df,state_features,policy, behav_pol,device))  # TODO Run weighted importance sampling with learned policy and behavior policy
        np.save(pol_eval_file_wis, evaluations_wis)
        #
        DR_agent, DR_gov, q_agent, action_agent = eval_policy_dr(val_df,state_features,policy, behav_pol,device)
        action_save_file_iter = action_iter_test_file + f'iteration_{training_iters}.npy'
        np.save(action_save_file_iter, action_agent)
        q_save_file_iter = Q_iter_test_file + f'iteration_{training_iters}.npy'
        np.save(q_save_file_iter, q_agent)
        #
        evaluations_dr.append([DR_agent, DR_gov])
        np.save(pol_eval_file_dr, evaluations_dr)
        #
        Q_RL_test_set.append(q_agent)
        np.save(Q_RL_test_file, Q_RL_test_set)
        #action_agent.cpu().detach().numpy()
        action_RL_test_set.append(action_agent)
        np.save(action_RL_test_file, action_RL_test_set)
        #
        torch.save({'policy_Q_function':policy.Q.state_dict(), 'policy_Q_target':policy.Q_target.state_dict()}, pol_file)
        training_iters += int(parameters["eval_freq"])
        #training_iters += 1
        print(f"Training iterations: {training_iters}")
    np.save(loss_train_file, loss_stat)
    
        
#policy = DQN(num_actions,len(state_features),device='cpu')

def eval_policy_wis(val_df,state_features,policy, behav_pol,device):
    rho_1_H = []
    state, action, reward, next_state, done, country = ReplayBuffer_test(val_df, state_features)
    country_idx = np.unique(np.array(country))
    V_WIS = []
    V_gov = []    
    for idx,entity in enumerate(country_idx):
        state_idx = state[np.array(country)==entity].to(device)
        action_idx = action[np.array(country)==entity]
        action_idx = torch.tensor(action_idx,dtype=torch.long).to(device)
        p_obs_traj = (F.softmax(behav_pol(state_idx),dim=-1)).gather(1,action_idx) #all this is for the obs action
        p_new_traj = (F.softmax(policy.Q(state_idx)[0], dim=-1)).gather(1,action_idx)
        ratio_prob = (np.array(p_new_traj.detach().cpu())/np.array(p_obs_traj.detach().cpu()))
        rho_1_H.append(np.clip(ratio_prob.prod(),1e-30, 1e4))
        w_H = np.array(rho_1_H).sum()
        reward_traj = reward[np.array(country)==entity]
        len_traj = reward_traj.shape[0]
        reward_disc = 0
        for t in range(len_traj):
            reward_disc += (0.99**t) * np.array(reward_traj[t])
        V_WIS.append( (reward_disc*(rho_1_H[idx]/w_H)) )
        V_gov.append(reward_disc)
    WIS_estimator_agent = np.array(V_WIS).sum()
    WIS_estimator_gov = np.array(V_gov).sum()/len(country_idx)
    print("---------------------------------------")
    print(f"Evaluation over the test set WIS RL: {WIS_estimator_agent:.3f}")
    print("---------------------------------------")
    print(f"Evaluation over the test set WIS GOV: {WIS_estimator_gov:.3f}")
    return WIS_estimator_agent, WIS_estimator_gov


    
#this evaluation also gives us the actions and Q in the test set
def eval_policy_dr(val_df,state_features,policy, behav_pol,device):
    # Compute V_H+1-t = Vst +rho_t (r_t + gamma V_H_t - Q(s,a)) is computed recursively. Starts from the end so V_0 = 0 and Vst is sum_a pi_1(a/s) R_hat(s,a)
    state, action, reward, next_state, done, country = ReplayBuffer_test(val_df, state_features)
    country_idx = np.unique(np.array(country))    
    num_episodes = len(country_idx)
    V_DR_list_agent = []
    V_DR_list_gov = []
    RL_Q = []
    RL_actions = []
    for idx,entity in enumerate(country_idx):
        state_idx = state[np.array(country)==entity].to(device)
        action_idx = action[np.array(country)==entity]
        action_idx = torch.tensor(action_idx,dtype=torch.long).to(device)
        p_obs_traj = (F.softmax(behav_pol(state_idx),dim=-1)).gather(1,action_idx) #all this is for the obs action
        p_new_traj = (F.softmax(policy.Q(state_idx)[0], dim=-1)).gather(1,action_idx)
        ratio_prob = p_new_traj.detach()/p_obs_traj.detach()
        reward_traj = reward[np.array(country)==entity].to(device)
        V_dr_agent = 0.0
        V_dr_gov = 0.0
        len_traj = state_idx.shape[0]
        for t in range(len_traj):
            V_st = (F.softmax(policy.Q(state_idx)[0][len_traj-1 - t,:]).detach()*policy.Q(state_idx)[0][len_traj-1 - t,:].detach()).sum()
            V_dr_agent = V_st + torch.tensor(ratio_prob[len_traj-1 - t])*( reward_traj[len_traj-1 - t]+ .99 * V_dr_agent - policy.Q(state_idx)[0][len_traj-1 - t,:].detach().gather(0,action_idx[len_traj-1 - t]) )
            V_dr_gov = V_st + torch.tensor(1)*( reward_traj[len_traj-1 - t]+ .99 * V_dr_gov - policy.Q(state_idx)[0][len_traj-1 - t,:].detach().gather(0,action_idx[len_traj-1 - t]) )
        V_DR_list_agent.append(np.clip(V_dr_agent.cpu(),-500, 500))
        V_DR_list_gov.append(V_dr_gov.cpu())
        q_agent, action_agent = policy.Q(state_idx)[0].max(1, keepdim=True)
        RL_Q.append(q_agent.cpu().detach().numpy())
        RL_actions.append(action_agent.cpu().detach().numpy())
    #then we need to average across the observed trajectories
    DR_agent = np.mean(V_DR_list_agent)
    DR_gov = np.mean(np.array(V_DR_list_gov))
    print("---------------------------------------")
    print(f"Evaluation over the test set DR RL: {DR_agent:.3f}")
    print("---------------------------------------")
    print(f"Evaluation over the test set DR GOV: {DR_gov:.3f}")
    return DR_agent, DR_gov, RL_Q, RL_actions



#policy_eval_save_file_wis = './srl_test/results/rewards_gap5/states_t/DQN_policy_eval_raw_data_wis'
#policy_eval_save_file_dr = './srl_test/results/rewards_gap5/states_t/DQN_policy_eval_raw_data_dr'
#policy_save_file = './srl_test/results/rewards_gap5/states_t/DQN_policy_raw_data'
 
def train_dBCQ_policy(train_df, val_df, state_dim, state_features, num_actions, batch_size, device, policy_save_file, policy_eval_save_file_wis,
    policy_eval_save_file_dr,loss_train_save_file, q_rl_save_file,action_rl_save_file ,behav_policy_file_wDemo):
    pol_learning_rate=1e-3
    params = {
        "eval_freq": 500,
        "discount": 0.99,
        #"batch_size": minibatch_size,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": pol_learning_rate
        },
        "max_timesteps": 5e5,
        "policy_file": policy_save_file+f'_lr{pol_learning_rate}.pt',
        "pol_eval_file_wis": policy_eval_save_file_wis+f'_lr{pol_learning_rate}.npy',
        "pol_eval_file_dr": policy_eval_save_file_dr+f'_lr{pol_learning_rate}.npy',
        "loss_train_file": loss_train_save_file+f'_lr{pol_learning_rate}.npy',
        "Q_RL_file": q_rl_save_file+f'_lr{pol_learning_rate}.npy',
        "action_RL_file": action_rl_save_file+f'_lr{pol_learning_rate}.npy',
        "action_iter_test": action_rl_save_file,
        "Q_iter_test": q_rl_save_file,         
    }
    behav_pol = FC_BC(state_dim, num_actions, 128).to(device)
    behav_pol.load_state_dict(torch.load(behav_policy_file_wDemo))          
    train_dBCQ(train_df,val_df,batch_size, num_actions, state_dim,state_features, device, params, behav_pol)


def run_dbcq():
    batch_size = 256
    # load the data
    df = pd.read_csv('C:/Users/pabgon/rl_representations/data/df_rewards_reproduction_rate_23March2022.csv', dtype='object', header=0)
    state_features = [ 'stringency_index', 'new_cases',
           'population_density', 'gdp_per_capita', 'diabetes_prevalence',
           'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
           'life_expectancy', 'hospital_beds_per_thousand' ]
    state_dim = len(state_features)
    num_actions= 5   
    context_dim= 8
    device='cuda'
    behav_policy_file_wDemo= './behavcloning/test/new_cases_gap5/BC_model.pt'
    # with rewards_5
    # Normalize some columns
    df_normalized = copy.deepcopy(df)
    temp=df_normalized[state_features].astype(float).describe()
    means=temp.loc['mean'].values
    stds=temp.loc['std'].values
    df_normalized[state_features] = (df_normalized[state_features].astype(float)-means)/stds
    df_normalized['reward'] = pd.to_numeric(df_normalized['reward_5'])
    #
    kf = KFold(n_splits=5, random_state=100, shuffle=True)
    idx_country = np.unique(df_normalized['Entity'])
    v_fold = 1
    for train_index, test_index in kf.split(idx_country):
        country_train, country_test = idx_country[train_index], idx_country[test_index]
        train_df = df_normalized[df_normalized['Entity'].isin(country_train)]
        val_df = df_normalized[df_normalized['Entity'].isin(country_test)]
        print(v_fold)
        train_df['action'] = pd.to_numeric(train_df['facial_coverings']) 
        val_df['action'] = pd.to_numeric(val_df['facial_coverings']) 
        countries_file = './srl_test/results/rewards_gap5/states_t/dBCQ/countries_test_set' + f'_rnd_state{v_fold}.npy'
        np.save(countries_file,np.unique(val_df['Entity']))
        #
        loss_train_save_file = './srl_test/results/rewards_gap5/states_t/dBCQ/loss_train' + f'_rnd_state{v_fold}'
        policy_eval_save_file_wis = './srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_eval_wis' + f'_rnd_state{v_fold}'
        policy_eval_save_file_dr = './srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_eval_dr' + f'_rnd_state{v_fold}'
        policy_save_file = './srl_test/results/rewards_gap5/states_t/dBCQ/dBCQ_policy_data' + f'_rnd_state{v_fold}'
        q_rl_save_file = './srl_test/results/rewards_gap5/states_t/dBCQ/Q_values' + f'_rnd_state{v_fold}'
        action_rl_save_file = './srl_test/results/rewards_gap5/states_t/dBCQ/action_chosen' + f'_rnd_state{v_fold}'
        #action_chosen_iter_save_file = './srl_test/results/rewards_gap5/states_t/action_chosen_iter' + f'_rnd_state{v_fold}'
        #
        train_dBCQ_policy(train_df,val_df,state_dim,state_features,num_actions,batch_size,device,policy_save_file, policy_eval_save_file_wis,policy_eval_save_file_dr,loss_train_save_file,q_rl_save_file,action_rl_save_file ,behav_policy_file_wDemo)
        v_fold += 1


if __name__ == '__main__':
    run_dbcq()
    

'''
for rnd_state in ([1,5,10,15,20]):  
        # split data into train, val
        idx_country = np.unique(df_normalized['Entity'])
        train_idx, val_idx = train_test_split(idx_country, test_size = 0.2, random_state = rnd_state)
        train_df = df_normalized[df_normalized['Entity'].isin(train_idx)]
        val_df = df_normalized[df_normalized['Entity'].isin(val_idx)]
'''
