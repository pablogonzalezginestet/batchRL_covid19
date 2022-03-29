"""
This script is used to develop a baseline policy using only the observed patient data via Behavior Cloning.

This baseline policy is then used to truncate and guide evaluation of policies learned using dBCQ. It should only need to be
run once for each unique cohort that one looks to learn a better treatment policy for.

The patient cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

# IMPORTS
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd

from dBCQ_utils import BehaviorCloning
from utils import ReplayBuffer
import copy
from sklearn.model_selection import train_test_split



def run(BC_network, train_dataloader, num_epochs, storage_dir, loss_func):
	# Construct training and validation loops
	#validation_losses = []
	training_losses = []
	training_iters = 0
	eval_frequency = 1

	for i_epoch in range(num_epochs):
		
		train_loss = BC_network.train_epoch(train_dataloader)
		training_losses.append(train_loss)

		if i_epoch % eval_frequency == 0:
			#eval_errors = []
			#BC_network.model.eval()
			#with torch.no_grad():
			#    for val_state, val_action in val_dataloader:
			#        val_state = val_state.to(torch.device('cuda'))
			#        val_action = val_action.to(torch.device('cuda'))
			#        pred_actions = BC_network.model(val_state)
			#        try:
			#            eval_loss = loss_func(pred_actions, val_action.flatten())
			#            eval_errors.append(eval_loss.item())
			#        except:
			#            print("LOL ERRORS")

			#mean_val_loss = np.mean(eval_errors)
			#validation_losses.append(mean_val_loss)
			#np.save(storage_dir+'validation_losses.npy', validation_losses)
			np.save(storage_dir+'training_losses.npy', training_losses)

			print(f"Training iterations: {i_epoch}, Training Loss: {training_losses}")
			# Save off and store trained BC model
			torch.save(BC_network.model.state_dict(), storage_dir+'BC_model.pt')

			BC_network.model.train()
	torch.save(BC_network.model.state_dict(), storage_dir+'BC_model.pt')
	print("Finished training Behavior Cloning model")
	print('+='*30)


class Discrete_RL_dataset(Dataset):
	"""
	Need the df Ready and scaled
	Normalization is Done before
	"""
	def __init__(self, df):
		self.df = df
	def __len__(self):
		return self.df.shape[0]
	def __getitem__(self,idx):
		temp=self.df.iloc[idx,:]
	# Rewards is always +- 15 at the terminal step and 
		done = float(temp['done'])
		#states=torch.FloatTensor(temp.iloc[6:12].values.astype(np.float32))#.to(device)
		states=torch.FloatTensor(temp[state_features].values.astype(np.float32))
		#assert states.shape==(7,)
		#if done:
		#	next_states=torch.zeros_like(states)#.to(device)
		#else:
		#	next_states=torch.FloatTensor(self.df[state_features].iloc[idx+1,:].astype(np.float32))#.to(device)
		#assert next_states.shape==(7,)
		#reward=float(temp['reward'])
		fc = int(float(temp['facial_coverings']))
		#shr = int(float(temp['stay_home_requirements']))
		#itc = int(float(temp['international_travel_controls']))
		#action = action_map[fc,shr]
		action = action_map[fc]
		#weight = temp['imp_weight']
		return states,action

if __name__ == '__main__':

	# Define input arguments and parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--demographics', dest='dem_context', default=True, action='store_true')
	parser.add_argument('--num_nodes', dest='num_nodes', default=128, type=int)
	parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4, type=float)
	parser.add_argument('--storage_folder', dest='storage_folder', default='test/new_cases_gap5', type=str)
	parser.add_argument('--batch_size', dest='batch_size', default=128, type=int)
	parser.add_argument('--num_epochs', dest='num_epochs', default=500, type=int)
	parser.add_argument('--weight_decay', dest='weight_decay', default=0.1, type=float)
	parser.add_argument('--optimizer_type', dest='optim_type', default='adam', type=str)

	args = parser.parse_args()

	device = torch.device('cuda')

	input_dim = 10 
	num_actions = 5
	if args.dem_context:
		train_buffer_file = './data/rewards_new_cases/gap_5/train_set_tuples' 
		#train_buffer_file = './data/rewards_new_cases/gap_14/train_set_tuples' 
		#validation_buffer_file = '/scratch/ssd001/home/tkillian/ml4h2020_srl/raw_data_buffers/val_buffer'
	else:
		train_buffer_file = './data/rewards_new_cases/gap_5/train_set_tuples' 
		#validation_buffer_file = '/scratch/ssd001/home/tkillian/ml4h2020_srl/raw_data_buffers/val_noCntxt_buffer'

	storage_dir = './behavcloning/' + args.storage_folder + '/'

	if not os.path.exists(storage_dir):
		os.mkdir(storage_dir)

	#df = pd.read_csv('C:\\Users\\pabgon\Documents\Project_Covid_mask_policy\data\df_final_nonpharma.csv', dtype='object', header=0)
	#df = pd.read_csv(r'C:/Users/pabgon/rl_representations/data/df_rewards_new_cases.csv', dtype='object', header=0)
	df = pd.read_csv(r'C:/Users/pabgon/rl_representations/data/df_rewards_new_cases_23_March.csv', dtype='object', header=0)
	# let s work with rewards_5
	df['reward'] = df['reward_5']

	state_features = [ 'new_cases', 'population_density', 'gdp_per_capita', 'diabetes_prevalence', 'cardiovasc_death_rate' , 'aged_65_older', 'human_development_index', 'life_expectancy', 'hospital_beds_per_thousand','stringency_index'  ]
	state_size = len(state_features)


	#group categories of actions
	#Face coverings
	#0- No policy
	#1- Recommended
	#2- Required 
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


	action_space_1 = np.unique(df['facial_coverings'])
	#action_space_2 = np.unique(df['stay_home_requirements'])
	#action_space_4 = np.unique(df['international_travel_controls'])

	action_map = {}
	count = 0
	for fm in range(len(action_space_1)):
		action_map[fm] = count
		count += 1

	action_train = []
	for idx,obser in train_df.iterrows():
		fc = pd.to_numeric(obser['facial_coverings'])
		action_train.append(action_map[fc])

	#action_map = {}
	#count = 0
	#for fm in range(len(action_space_1)):
	#	for shr in range(len(action_space_2)):
	#			action_map[(fm,shr)] = count
	#			count += 1
   
	#action_train = []
	#for idx,obser in train_df.iterrows():
	#			fc = pd.to_numeric(obser['facial_coverings'])
	#			shr = pd.to_numeric(obser['stay_home_requirements'])
	#			#itc = pd.to_numeric(obser['international_travel_controls'])
	#			action_train.append(action_map[fc,shr])

	train_df['action'] = action_train 
	ds=Discrete_RL_dataset(train_df)
	train_dataloader=DataLoader(ds,batch_size=64,shuffle=True)
	'''
	val_buffer = ReplayBuffer(input_dim, args.batch_size, 50000, device)
	val_buffer.load(validation_buffer_file)
	val_states = val_buffer.state[:val_buffer.crt_size]
	val_actions = val_buffer.action[:val_buffer.crt_size]
	val_dataset = TensorDataset(torch.from_numpy(val_states).float(), torch.from_numpy(val_actions).long())
	val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
	'''
	# Initialize the BC network
	BC_network = BehaviorCloning(input_dim, num_actions, args.num_nodes, args.learning_rate, args.weight_decay, args.optim_type, device)

	loss_func = nn.CrossEntropyLoss()

	run(BC_network, train_dataloader, args.num_epochs, storage_dir, loss_func)