### Plot Reproduction rate vs reward function, k=5,14

import matplotlib
import matplotlib.pyplot as plt

import random
import os
import sys
import pickle
#import click
import yaml
import numpy as np
from experiment import Experiment

import torch

import numpy as np
import pandas as pd
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from itertools import chain

import matplotlib.dates as mdates

#plot rewards and new cases

df = pd.read_csv('C:/Users/pabgon/rl_representations/data/df_rewards_reproduction_rate_23March2022.csv', dtype='object', header=0)
state_features = [ 'stringency_index', 'new_cases',
       'population_density', 'gdp_per_capita', 'diabetes_prevalence',
       'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
       'life_expectancy', 'hospital_beds_per_thousand' ]
state_dim = len(state_features)
num_actions= 5

countries = np.unique(df['Entity'])
for c in range(len(countries)):
    country = countries[c]
    reward = np.array(df[df['Entity']== country]['reward_5'],dtype=float)
    R = np.array(df[df['Entity']== country]['reproduction_rate'],dtype=float)
    day = df[df['Entity']== country]['Day']
    fig,ax = plt.subplots()
    ax = plt.gca()
    locator = mdates.DayLocator(interval=50)
    ax.xaxis.set_major_locator(locator)
    ax.plot(day, R , color="red", marker="o")
    ax.set_xlabel("time",fontsize=14)
    ax.set_ylabel("Reproduction rate",fontsize=14)
    ax2=ax.twinx()
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(locator)
    ax2.plot(day, reward,color="blue",marker="o", label="reward")
    ax2.set_ylabel("Reward",fontsize=14)
    ax2.set_title(country)
    # save the plot as a file
    fig.set_size_inches((8.5, 11), forward=False)
    fig.savefig('./srl_test/results/rewards_gap5/plot_rewards/{}.png'.format(country),format='png',dpi=500)
    plt.close(fig)
    
countries = np.unique(df['Entity'])
for c in range(len(countries)):
    country = countries[c]
    reward14 = np.array(df[df['Entity']== country]['reward_14'],dtype=float)
    reward5 = np.array(df[df['Entity']== country]['reward_5'],dtype=float)
    R = np.array(df[df['Entity']== country]['reproduction_rate'],dtype=float)
    day = df[df['Entity']== country]['Day']
    fig,ax = plt.subplots()
    ax = plt.gca()
    locator = mdates.DayLocator(interval=50)
    ax.xaxis.set_major_locator(locator)
    ax.plot(day, R , color="red", marker="o")
    ax.set_xlabel("time",fontsize=14)
    ax.set_ylabel("Reproduction rate",fontsize=14)
    ax2=ax.twinx()
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(locator)
    ax2.plot(day, reward5,color="blue",marker="o", label="reward k=5")
    ax2.plot(day, reward14,color="green",marker="o", label="reward k=14")
    ax2.set_ylabel("Rewards: k=5 (blue) and k=14 (green)",fontsize=14)
    ax2.set_title(country)
    # save the plot as a file
    fig.set_size_inches((8.5, 11), forward=False)
    fig.savefig('./srl_test/results/rewards_gap14/plot_rewards/{}.png'.format(country),format='png',dpi=500)
    plt.close(fig)  

countries = np.unique(df['Entity'])

for c in range(len(countries)):
    country = countries[c]
    reward14 = np.array(df[df['Entity']== country]['reward_14'],dtype=float)
    reward5 = np.array(df[df['Entity']== country]['reward_5'],dtype=float)
    R = np.array(df[df['Entity']== country]['reproduction_rate'],dtype=float)
    day = df[df['Entity']== country]['Day']
    day = pd.to_datetime(day, format='%Y-%m-%d')
    from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter
    months = MonthLocator(range(1, 13), bymonthday=1, interval=2)
    monthsFmt = DateFormatter("%b '%y")
    fig,ax = plt.subplots()
    ax = plt.gca()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    #date_form = mdates.DateFormatter("%b-%Y")
    #locator = mdates.DayLocator(interval=60)
    #ax.xaxis.set_major_locator(locator)
    #ax.xaxis.set_major_formatter(date_form)
    ax.plot(day, R , color="red", marker="o")
    ax.set_xlabel("",fontsize=20)
    ax.set_ylabel("Reproduction rate",fontsize=20)
    #for label in ax.get_xticklabels(which='major'):
    #    label.set(rotation=30, horizontalalignment='right')
    ax2=ax.twinx()
    ax2 = plt.gca()
    ax2.xaxis.set_major_locator(months)
    ax2.xaxis.set_major_formatter(monthsFmt)
    ax2.plot(day, reward5,color="blue",marker="o", label="reward k=5")
    ax2.plot(day, reward14,color="green",marker="o", label="reward k=14")
    #for label in ax2.get_xticklabels(which='major'):
    #    label.set(rotation=30, horizontalalignment='right')
    ax2.set_ylabel("Rewards: k=5 (blue) and k=14 (green)",fontsize=20)
    ax2.set_yticks([-1,0,1])
    #ax2.set_title(country)
    # save the plot as a file
    fig.set_size_inches((8.27, 12.5))
    fig.savefig('./srl_test/results/plot_rewards/{}.pdf'.format(country),format='pdf',dpi=500,orientation='portrait')
    plt.close(fig) 

    matplotlib.dates.MonthLocator(interval=6)
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax2.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')    
