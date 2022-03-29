'''
This script preprocesses and organizes the Sepsis patient cohort extracted with the procedure 
provided at: https://github.com/microsoft/mimic_sepsis to produce patient trajectories for easier
use in sequential models.

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

'''

import sys
import os
import time
import torch

import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import train_test_split

save_dir = 'data/rewards_new_cases/gap_5'
train_file = 'train_set_tuples'
val_file = 'val_set_tuples'
test_file = 'test_set_tuples'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cuda' if torch.cuda.is_available() else 'cpu'

#df = pd.read_csv('C:\\Users\\pabgon\Documents\Project_Covid_mask_policy\data\df_final_nonpharma.csv', dtype='object', header=0)
#df = pd.read_csv(r'C:/Users/pabgon/rl_representations/data/df_rewards_new_cases.csv', dtype='object', header=0)
df = pd.read_csv(r'C:/Users/pabgon/rl_representations/data/df_rewards_new_cases_23_March.csv', dtype='object', header=0)

# let s work with rewards_5
df['reward'] = df['reward_5']

# states variables

state_features = [ 'new_cases', 'population_density', 'gdp_per_capita', 'diabetes_prevalence', 'cardiovasc_death_rate' , 'aged_65_older', 'human_development_index', 'life_expectancy', 'hospital_beds_per_thousand','stringency_index'  ]
#state_features = [ 'new_cases_per_million', 'population_density', 'gdp_per_capita', 'diabetes_prevalence' ,'reproduction_rate', 'aged_65_older' ]
state_size = len(state_features)

#group categories of actions
#Face coverings
 
df['facial_coverings'] = df['facial_coverings'].astype(float)
#df['facial_coverings'][np.array(df['facial_coverings'],dtype=float)>=2] =2

#Stay at home:
#0 - No measures
#1 - recommend not leaving house
#2 - require
#df['stay_home_requirements'] = df['stay_home_requirements'].astype(float)
#df['stay_home_requirements'][np.array(df['stay_home_requirements'],dtype=float)>=2] = 2

# Normalize some columns
df_normalized = copy.deepcopy(df)
temp=df_normalized[state_features].astype(float).describe()
means=temp.loc['mean'].values
stds=temp.loc['std'].values
df_normalized[state_features] = (df_normalized[state_features].astype(float)-means)/stds
df_normalized['reward'] = pd.to_numeric(df_normalized['reward'])

# split data into train, val
idx_country = np.unique(df_normalized['Entity'])
train_idx, val_idx = train_test_split(idx_country, test_size = 0.2, random_state = 5)
train_df = df_normalized[df_normalized['Entity'].isin(train_idx)]
val_df = df_normalized[df_normalized['Entity'].isin(val_idx)]

#####################################################################################
action_space_1 = np.unique(df['facial_coverings'])
#action_space_2 = np.unique(df['stay_home_requirements'])
#action_space_4 = np.unique(df['international_travel_controls'])

action_map = {}
count = 0
for fm in range(len(action_space_1)):
            action_map[fm] = count
            count += 1

'''
action_map = {}
count = 0
#for fm in range(len(action_space_1)):
#    for shr in range(len(action_space_2)):
#            action_map[(fm,shr)] = count
#            count += 1

#for fm in range(len(action_space_1)):
#    for shr in range(len(action_space_2)):
#            for itc in range(len(action_space_4)):
#                action_map[(fm,shr,itc)] = count
#                count += 1

#action_train = []
#for idx,obser in train_df.iterrows():
#            fc = pd.to_numeric(obser['facial_coverings'])
#            shr = pd.to_numeric(obser['stay_home_requirements'])
#            action_train.append(action_map[fc,shr])
            #itc = pd.to_numeric(obser['international_travel_controls'])
            #action_train.append(action_map[fc,shr,itc])
#train_df['action'] = action_train  
 
action_test = []
for idx,obser in val_df.iterrows():
            fc = pd.to_numeric(obser['facial_coverings'])
            shr = pd.to_numeric(obser['stay_home_requirements'])
            action_test.append(action_map[fc,shr])
            #itc = pd.to_numeric(obser['international_travel_controls'])
            #action_test.append(action_map[fc,shr,itc])
val_df['action'] = action_test 
'''
action_train = []
for idx,obser in train_df.iterrows():
            fc = pd.to_numeric(obser['facial_coverings'])
            action_train.append(action_map[fc])

train_df['action'] = action_train 

action_test = []
for idx,obser in val_df.iterrows():
            fc = pd.to_numeric(obser['facial_coverings'])
            action_test.append(action_map[fc])

val_df['action'] = action_test 

# Define the features of the full data



device = 'cpu'


################################################################
#          FORMAT DATA FOR USE IN SEQUENTIAL MODELS
################################################################
dem_keep_cols = ['population_density', 'gdp_per_capita','diabetes_prevalence', 'cardiovasc_death_rate', 'aged_65_older', 'human_development_index', 'life_expectancy','hospital_beds_per_thousand']
#dem_keep_cols = ['population_density', 'gdp_per_capita','diabetes_prevalence',  'aged_65_older']
#obs_keep_cols = ['new_cases_per_million', 'reproduction_rate' ]
obs_keep_cols = ['new_cases', 'stringency_index' ]

num_actions = len(action_map)
num_obs = len(obs_keep_cols)
#num_dem = 4
num_dem = len(dem_keep_cols) # 

dem_cols = [i for i in train_df.columns if i in dem_keep_cols]
obs_cols = [i for i in train_df.columns if i in obs_keep_cols]
ac_col  = 'action'
rew_col = 'reward'

## look at the orden obs_cols ['stringency_index', 'new_cases']
## TRAINING DATA
#---------------------------------------------------------------------------
print("Converting Training Data")
print("--"*20)
#train_data[ac_col] = train_data[ac_col]
#all_actions = train_data[ac_col].unique()
#all_actions.sort()
#try:
    #all_actions = all_actions.astype(np.int32)
#except:
#    raise ValueError('Actions are expected to be integers, but are not.')
# if not all(all_actions == np.arange(num_actions, dtype=np.int32)):
    # print(Font.red + 'Some actions are missing from data or all action space not properly defined.' + Font.end)
data_trajectory = {}
data_trajectory['dem_cols'] = dem_cols
data_trajectory['obs_cols'] = obs_cols
data_trajectory['ac_col']  = 'action'
data_trajectory['rew_col'] = 'reward'
data_trajectory['num_actions'] = num_actions
data_trajectory['obs_dim'] = len(obs_cols)
data_trajectory['traj'] = {}
data_trajectory['pos_traj'] = []
data_trajectory['neg_traj'] = []
data_trajectory['done'] = 'done'

trajectories = train_df['Entity'].unique()
for i in trajectories:
    # bar.update()
    traj_i = train_df[train_df['Entity'] == i].sort_values(by='Day')
    #traj_j = train_acuity[train_acuity['traj']==i].sort_values(by='step')
    data_trajectory['traj'][i] = {}
    data_trajectory['traj'][i]['dem'] = torch.Tensor(traj_i[dem_cols].values).to('cpu')
    data_trajectory['traj'][i]['obs'] = torch.Tensor(traj_i[obs_cols].values).to('cpu')
    data_trajectory['traj'][i]['actions'] = torch.Tensor(traj_i['action'].values.astype(np.int32)).to('cpu').long()
    data_trajectory['traj'][i]['rewards'] = torch.Tensor(traj_i['reward'].values).to('cpu')
    data_trajectory['traj'][i]['done'] = torch.Tensor(traj_i['done'].values.astype(np.int32)).to('cpu')
    #data_trajectory['traj'][i]['acuity'] = torch.Tensor(traj_j[acuity_cols].values).to('cpu')
    if sum(traj_i['reward'].values) > 0:
        data_trajectory['pos_traj'].append(i)
    else:
        data_trajectory['neg_traj'].append(i)

horizon_max = []
for ii, traj in enumerate(trajectories):
    #obs = data_trajectory['traj'][traj]['obs']
    horizon_max.append(data_trajectory['traj'][traj]['dem'].shape[0])
    
horizon = np.max(horizon_max)
observations = torch.zeros((len(trajectories), horizon, num_obs))
demographics = torch.zeros((len(trajectories), horizon, num_dem)) 
#actions = torch.zeros((len(trajectories), horizon-1, num_actions))
actions = torch.zeros((len(trajectories), horizon, num_actions))
lengths = torch.zeros((len(trajectories)), dtype=torch.int)
times = torch.zeros((len(trajectories), horizon))
rewards = torch.zeros((len(trajectories), horizon))
dones = torch.zeros((len(trajectories), horizon))
#acuities = torch.zeros((len(trajectories), horizon-1, num_acuity_scores))
action_temp = torch.eye(num_actions)
for ii, traj in enumerate(trajectories):
    obs = data_trajectory['traj'][traj]['obs']
    dem = data_trajectory['traj'][traj]['dem']
    action = data_trajectory['traj'][traj]['actions'].view(-1,1)
    reward = data_trajectory['traj'][traj]['rewards']
    done = data_trajectory['traj'][traj]['done']
    length = dem.shape[0]
    lengths[ii] = length
    temp = action_temp[action].squeeze(1)
    #observations[ii] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
    if horizon != length:
        observations[ii] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
        demographics[ii] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
        actions[ii] = torch.cat((temp, torch.zeros((horizon-length, num_actions), dtype=torch.float)))
        rewards[ii] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))
        dones[ii] = torch.cat((done, torch.zeros((horizon-length), dtype=torch.float)))
    elif horizon == length:
        observations[ii] = obs
        demographics[ii] = dem
        actions[ii] = temp
        rewards[ii] = reward
        dones[ii] = done
    times[ii] = torch.Tensor(range(horizon))
    
    #acuities[ii] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))

# Eliminate single transition trajectories...
actions = actions[lengths>1.0].to(device)
observations = observations[lengths>1.0].to(device)
demographics = demographics[lengths>1.0].to(device)
times = times[lengths>1.0].to(device)
rewards = rewards[lengths>1.0].to(device)
#acuities = acuities[lengths>1.0].to(device)
lengths = lengths[lengths>1.0].to(device)


## Test DATA
#---------------------------------------------------------------------------
print("Converting Test Data")
print("+"*20)
test_data_trajectory = {}
test_data_trajectory['obs_cols'] = obs_cols
test_data_trajectory['dem_cols'] = dem_cols
test_data_trajectory['ac_col']  = 'action'
test_data_trajectory['rew_col'] = 'reward'
test_data_trajectory['num_actions'] = num_actions
test_data_trajectory['obs_dim'] = len(obs_cols)
test_data_trajectory['traj'] = {}
test_data_trajectory['pos_traj'] = []
test_data_trajectory['neg_traj'] = []
test_data_trajectory['done'] = 'done'

test_trajectories = val_df['Entity'].unique()
for j in test_trajectories:
    traj_j = val_df[val_df['Entity']==j].sort_values(by='Day')
    test_data_trajectory['traj'][j] = {}
    test_data_trajectory['traj'][j]['obs'] = torch.Tensor(traj_j[obs_cols].values).to('cpu')
    test_data_trajectory['traj'][j]['dem'] = torch.Tensor(traj_j[dem_cols].values).to('cpu')
    test_data_trajectory['traj'][j]['actions'] = torch.Tensor(traj_j['action'].values.astype(np.int32)).to('cpu').long()
    test_data_trajectory['traj'][j]['rewards'] = torch.Tensor(traj_j['reward'].values).to('cpu')
    test_data_trajectory['traj'][j]['done'] = torch.Tensor(traj_j['done'].values.astype(np.int32)).to('cpu') 
    #test_data_trajectory['traj'][j]['acuity'] = torch.Tensor(traj_k[acuity_cols].values).to('cpu')
    if sum(traj_j['reward'].values) > 0:
        test_data_trajectory['pos_traj'].append(j)
    else:
        test_data_trajectory['neg_traj'].append(j)
         
horizon_max_test = []
for ii, traj in enumerate(test_trajectories):
    #obs = data_trajectory['traj'][traj]['obs']
    horizon_max_test.append(test_data_trajectory['traj'][traj]['dem'].shape[0])

horizon_test = np.max(horizon_max_test)    
test_obs = torch.zeros((len(test_trajectories), horizon_test, num_obs))
test_dem = torch.zeros((len(test_trajectories), horizon_test, num_dem))
test_actions = torch.zeros((len(test_trajectories), horizon_test, num_actions))
test_lengths = torch.zeros((len(test_trajectories)), dtype=torch.int)
test_times = torch.zeros((len(test_trajectories), horizon_test))
test_rewards = torch.zeros((len(test_trajectories), horizon_test))
test_dones = torch.zeros((len(test_trajectories), horizon_test))

action_temp = torch.eye(num_actions)
for jj, traj in enumerate(test_trajectories):
    obs = test_data_trajectory['traj'][traj]['obs']
    dem = test_data_trajectory['traj'][traj]['dem']
    action = test_data_trajectory['traj'][traj]['actions'].view(-1,1)
    reward = test_data_trajectory['traj'][traj]['rewards']
    done = test_data_trajectory['traj'][traj]['done']
    length = dem.shape[0]
    test_lengths[jj] = length
    temp = action_temp[action].squeeze(1)
    if horizon_test != length:  
        test_obs[jj] = torch.cat((obs, torch.zeros((horizon_test-length, obs.shape[1]), dtype=torch.float)))
        test_dem[jj] = torch.cat((dem, torch.zeros((horizon_test-length, dem.shape[1]), dtype=torch.float)))
        test_actions[jj] = torch.cat((temp, torch.zeros((horizon_test-length, num_actions), dtype=torch.float)))
        test_rewards[jj] = torch.cat((reward, torch.zeros((horizon_test-length), dtype=torch.float)))
        test_dones[jj] = torch.cat((done, torch.zeros((horizon_test-length), dtype=torch.float)))
    elif horizon_test == length:
        test_obs[jj] = obs
        test_dem[jj] = dem
        test_actions[jj] = temp
        test_rewards[jj] = reward
        test_dones[jj] = done
    test_times[jj] = torch.Tensor(range(horizon_test))

# Eliminate single transition trajectories...
test_actions = test_actions[test_lengths>1.0].to(device)
test_obs = test_obs[test_lengths>1.0].to(device)
test_dem = test_dem[test_lengths>1.0].to(device)
test_times = test_times[test_lengths>1.0].to(device)
test_rewards = test_rewards[test_lengths>1.0].to(device)
test_lengths = test_lengths[test_lengths>1.0].to(device)



#### Save off the tuples...
#############################
print("Saving off tuples")
print("..."*20)
torch.save((demographics,observations,actions,lengths,times,rewards, dones),os.path.join(save_dir,train_file))

#torch.save((val_dem,val_obs,val_actions,val_lengths,val_times,val_rewards),os.path.join(save_dir,val_file))

torch.save((test_dem,test_obs,test_actions,test_lengths,test_times,test_rewards, test_dones),os.path.join(save_dir,test_file))

print("\n")
print("Finished conversion")

## the orden of obs is obs_cols ['stringency_index', 'new_cases']
'''
# We also extract and save off the mortality outcome of patients in the test set for evaluation and analysis purposes
print("\n")
print("Extracting Test set mortality")
test_mortality = torch.Tensor(test_data.groupby('traj')['r:reward'].sum().values)
test_mortality = test_mortality.unsqueeze(1).unsqueeze(1)
test_mortality = test_mortality.repeat(1,20,1)  # Put in the same general format as the patient trajectories
# Save off mortality tuple
torch.save(test_mortality,os.path.join(save_dir,'test_mortality_tuple'))
'''
'''
>>> val_df['Entity'].unique()
array(['Barbados', 'Belgium', 'Bulgaria', 'Burundi', 'Cameroon', 'Egypt',
       'Eritrea', 'France', 'Gabon', 'Gambia', 'Ghana', 'Haiti',
       'Iceland', 'India', 'Iran', 'Iraq', 'Kenya', 'Latvia', 'Lebanon',
       'Luxembourg', 'Nepal', 'Netherlands', 'New Zealand', 'Serbia',
       'South Korea', 'Switzerland', 'Timor', 'Trinidad and Tobago',
       'Venezuela'], dtype=object)
>>> train_df['Entity'].unique()
array(['Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Australia',
       'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
       'Belarus', 'Belize', 'Benin', 'Bhutan', 'Bolivia',
       'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei',
       'Burkina Faso', 'Cambodia', 'Canada', 'Cape Verde',
       'Central African Republic', 'Chile', 'China', 'Colombia',
       'Costa Rica', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
       'Djibouti', 'Dominican Republic', 'Ecuador', 'El Salvador',
       'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'Georgia',
       'Germany', 'Greece', 'Guatemala', 'Guyana', 'Honduras', 'Hungary',
       'Indonesia', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan',
       'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Liberia',
       'Libya', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mali',
       'Malta', 'Mexico', 'Moldova', 'Mongolia', 'Morocco', 'Mozambique',
       'Myanmar', 'Nicaragua', 'Niger', 'Norway', 'Oman', 'Pakistan',
       'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
       'Qatar', 'Romania', 'Russia', 'Saudi Arabia', 'Singapore',
       'Slovakia', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka',
       'Sudan', 'Suriname', 'Sweden', 'Tajikistan', 'Tanzania',
       'Thailand', 'Togo', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine',
       'United Arab Emirates', 'United Kingdom', 'United States',
       'Uruguay', 'Uzbekistan', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'],
      dtype=object)
'''