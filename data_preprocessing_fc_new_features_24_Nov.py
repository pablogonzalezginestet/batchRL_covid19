import json
import numpy as np
import pandas as pd

df_states = pd.read_csv('C:/Users/pabgon/Documents/Project_Covid_mask_policy/data/owid-covid-data.csv',) 

df_policy_fm = pd.read_csv('C:/Users/pabgon/Documents/Project_Covid_mask_policy/data/face-covering-policies-covid.csv',)  

df_stringency_index = pd.read_csv('C:/Users/pabgon/Documents/Project_Covid_mask_policy/data/covid-stringency-index.csv',) 

#df_policy_lockdown = pd.read_csv('C:/Users/pabgon/Documents/Project_Covid_mask_policy/data/stay-at-home-covid.csv',) 

merge_fm_st = pd.merge(df_policy_fm, df_stringency_index, how="right", on=["Entity", "Day"])
#merge_fm_st_lockdown = pd.merge(merge_fm_st,df_policy_lockdown, how="right", on=["Entity", "Day"])
merge_fm_st.info()
#merge_fm_st.shape
#merge_fm_st_lockdown.info()

# let s remove code_x code_y
#merge_fm_st_lockdown.drop(['Code_x', 'Code_y','Code'], axis=1,inplace=True)
merge_fm_st.drop(['Code_x', 'Code_y'], axis=1,inplace=True)

#merge_fm_st_lockdown['stringency_index'].isnull().sum()
#merge_fm_st_lockdown['facial_coverings'].isnull().sum()

#merge_fm_st.dropna(subset = ["facial_coverings"], inplace=True)
#ds_fm_st_ld = merge_fm_st_lockdown[~merge_fm_st_lockdown['facial_coverings'].isnull()]
ds_fm_st = merge_fm_st[~merge_fm_st['facial_coverings'].isnull()]


df_states.rename(columns = {'location':'Entity', 'date':'Day'}, inplace=True)

# Now we merge
#merge_all_df = pd.merge(df_states, ds_fm_st_ld, how="right", on=["Entity", "Day"])
merge_all_df = pd.merge(df_states, ds_fm_st, how="right", on=["Entity", "Day"])

# look at the description above and try to take states that have at least 60000 observations
merge_all_df.info()

merge_all_df = merge_all_df.rename(columns = {"stringency_index_y": "stringency_index"})
# To start
# state = new_cases_per_million
# action = face_coverings , stay_home_requirements, school_closures , international_travel_controls
# rewards = 1 if new cases is 0 and -1 if new cases is positive
#df_final = merge_all_df[ ['Entity'] + ['Day'] + 
#               ['facial_coverings'] + ['stay_home_requirements']   + ['human_development_index']  +
#               ['new_cases'] +['population_density'] + ['gdp_per_capita'] + ['diabetes_prevalence' ] + ['cardiovasc_death_rate' ] +  
#               ['reproduction_rate'] + ['aged_65_older'] + ['human_development_index'] + ['life_expectancy'] + ['hospital_beds_per_thousand'] + ['extreme_poverty']
#            ]


df_final = merge_all_df[ ['Entity'] + ['Day'] + 
               ['facial_coverings']   + 
               ['stringency_index']   + 
               ['new_cases'] +['population_density'] + ['gdp_per_capita'] + ['diabetes_prevalence' ] + ['cardiovasc_death_rate' ] +  
                ['aged_65_older'] + ['human_development_index'] + ['life_expectancy'] + ['hospital_beds_per_thousand'] 
            ]

df_final = merge_all_df[ ['Entity'] + ['Day'] + 
               ['facial_coverings']   +  ['stringency_index']   + 
               ['new_cases']  +['population_density'] + ['gdp_per_capita'] + ['diabetes_prevalence' ] + ['cardiovasc_death_rate' ] +  
                ['aged_65_older'] + ['human_development_index'] + ['life_expectancy'] + ['hospital_beds_per_thousand'] + ['reproduction_rate'] 
            ]
#['hospital_beds_per_thousand'] this variable is static
df_final.info()
#################################
# as another example I can include ICU variable  for a subset of countries those for that variable is available. We would have less countries
#################################

#accommodate the dates
date_f_l = df_final[['Entity','Day']].sort_values(by=['Entity','Day']).groupby('Entity').agg(['first','last'])
date_states_f_l = df_states[['Entity','Day']].sort_values(by=['Entity','Day']).groupby('Entity').agg(['first','last'])
date_policy_f_l = ds_fm_st[['Entity','Day']].sort_values(by=['Entity','Day']).groupby('Entity').agg(['first','last'])

lb_ub_dates = []
for i in range(len(date_f_l)):
    country_idx = date_f_l.index[i]
    f_l_temp_pol = date_policy_f_l[date_policy_f_l.index==country_idx]
    f_l_temp_st = date_states_f_l[date_states_f_l.index==country_idx]
    if (len(f_l_temp_st['Day']['first'])==1 and len(f_l_temp_pol['Day']['first'])==1):        
        lb = np.where(f_l_temp_pol['Day']['first']>=f_l_temp_st['Day']['first'],f_l_temp_pol['Day']['first'],f_l_temp_st['Day']['first'])
        ub = np.where(f_l_temp_pol['Day']['last']>=f_l_temp_st['Day']['last'],f_l_temp_st['Day']['last'],f_l_temp_pol['Day']['last'])
        lb_ub_dates.append(np.hstack([country_idx,lb,ub]))

df_final_subset = []
for i in range(len(lb_ub_dates)):
    df_final_temp = df_final[df_final['Entity']==lb_ub_dates[i][0]]
    df_final_subset.append(df_final_temp[( df_final_temp['Day'] >= lb_ub_dates[i][1]) & (df_final_temp['Day'] <= lb_ub_dates[i][2] )] )

df_final_processed = pd.concat(df_final_subset)
df_final_processed.info()

#i do not use reproduction rate in the model
#df_final_processed.dropna(subset=['reproduction_rate']).info()

np.unique(df_final_processed['Entity'][df_final_processed['gdp_per_capita'].isnull()])

np.unique(df_final_processed['Entity'][df_final_processed['diabetes_prevalence'].isnull()])

np.unique(df_final_processed['Entity'][df_final_processed['aged_65_older'].isnull()])

np.unique(df_final_processed['Entity'][df_final_processed['new_cases_per_million'].isnull()])

np.unique(df_final_processed['Entity'][df_final_processed['population_density'].isnull()])

#filter_rows_by_values(d,"str", ["b","c"])
def filter_rows_by_values(df, col, values):
    return df[df[col].isin(values) == False]

'''
I have removed the following countries (due to missing values or remote islands)
'Andorra', 
'Cuba', 
'Liechtenstein', 
'Monaco', 
'Somalia', 
'Syria',
'Taiwan',
'Kosovo',
'San Marino',
'Dominica',
'South Sudan',
'Vanuatu',
'Guinea',
'Solomon Islands',
'Mauritius',
'Seychelles',
'Faeroe Islands',
'Kiribati'
'''

df_final_rl = filter_rows_by_values(df_final_processed,'Entity',['Andorra', 
'Cuba', 
'Liechtenstein', 
'Monaco', 
'Somalia', 
'Syria',
'Taiwan',
'Kosovo',
'San Marino',
'Dominica',
'South Sudan',
'Vanuatu',
'Guinea',
'Solomon Islands',
'Mauritius',
'Seychelles',
'Faeroe Islands',
'Kiribati',
'Angola', 'Chad', 'Congo', "Cote d'Ivoire",
'Democratic Republic of Congo', 'Lesotho', 'Mauritania', 'Namibia',
       'Nigeria', 'Palestine', 'Papua New Guinea', 'Rwanda', 'Senegal',
       'Sierra Leone'])


#df_final_rl_update = df_final_rl.dropna(subset=['reproduction_rate'])

df_final_rl.info()

#>>> df_final_rl[df_final_rl['hospital_beds_per_thousand'].isna()]['Entity'].unique()
#array(['Angola', 'Chad', 'Congo', "Cote d'Ivoire",
#       'Democratic Republic of Congo', 'Lesotho', 'Mauritania', 'Namibia',
#       'Nigeria', 'Palestine', 'Papua New Guinea', 'Rwanda', 'Senegal',
#       'Sierra Leone'], dtype=object)

# remove the new cases that are negative
df_final_rl = df_final_rl[(df_final_rl['new_cases']>=0)]

# so we remove those countries that do not have data on the variable "hospital bed per thousand"
df_final_rl_update=df_final_rl.dropna(subset=['hospital_beds_per_thousand'])
#df_final_rl_update[df_final_rl_update['hospital_beds_per_thousand'].isna()]

#df_final_rl_update[df_final_rl_update['new_cases_per_million'].isna()]
#df_final_rl_update[df_final_rl_update['new_cases'].isna()]

df_final_rl_update.describe()

'''
>>> df_final_rl_update.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 65559 entries, 34 to 92038
Data columns (total 13 columns):
 #   Column                      Non-Null Count  Dtype
---  ------                      --------------  -----
 0   Entity                      65559 non-null  object
 1   Day                         65559 non-null  object
 2   facial_coverings            65559 non-null  float64
 3   stringency_index            65559 non-null  float64
 4   new_cases                   65559 non-null  float64
 5   population_density          65559 non-null  float64
 6   gdp_per_capita              65559 non-null  float64
 7   diabetes_prevalence         65559 non-null  float64
 8   cardiovasc_death_rate       65559 non-null  float64
 9   aged_65_older               65559 non-null  float64
 10  human_development_index     65559 non-null  float64
 11  life_expectancy             65559 non-null  float64
 12  hospital_beds_per_thousand  65559 non-null  float64
dtypes: float64(11), object(2)
memory usage: 7.0+ MB
'''

# so let s check how many observation per country
count_per_country = df_final_rl_update[['Entity','Day']].groupby('Entity').agg(['count'])
count_per_country[np.array(count_per_country<365)] # countries with less than 1 year of trajectory
#countries_with_less_than_year = np.array(count_per_country[np.array(count_per_country<365)].index)
#df_final_rl_update_1 = filter_rows_by_values(df_final_rl_update,'Entity',countries_with_less_than_year)
#df_final_rl_update_1.info() # this is the final data to work on

df_final_rl_update_1 = df_final_rl_update

# REWARDS
# incubation is 5 to 14 days based on CDC
# https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html#:~:text=The%20incubation%20period%20for%20COVID,from%20exposure%20to%20symptoms%20onset.&text=One%20study%20reported%20that%2097.5,SARS%2DCoV%2D2%20infection.
# using new cases per million
# let s define again the rewards for the nonpharma df using new cases
df_final_nonpharma_temp = df_final_rl_update_1[df_final_rl_update_1['Day'] <= '2021-04-30'] #with more days than 2021-03-31 in order to have data of the last seven values for each country
df_final_nonpharma_temp['reward_5'] = None
df_final_nonpharma_temp['reward_14'] = None
for i in range(len(df_final_nonpharma_temp)):
    if i+5<len(df_final_nonpharma_temp):        
        if df_final_nonpharma_temp['Entity'].iloc[i] == df_final_nonpharma_temp['Entity'].iloc[i+5]:
                s_today =  df_final_nonpharma_temp['new_cases'].iloc[i]
                s_next_week =  df_final_nonpharma_temp['new_cases'].iloc[i+5]
                if np.abs(s_next_week-s_today)>0:
                    df_final_nonpharma_temp['reward_5'].iloc[i] = - np.sign(s_next_week-s_today)
                else:
                    df_final_nonpharma_temp['reward_5'].iloc[i] = 1
    if i+14<len(df_final_nonpharma_temp): 
        if df_final_nonpharma_temp['Entity'].iloc[i] == df_final_nonpharma_temp['Entity'].iloc[i+14]:
            s_today_ =  df_final_nonpharma_temp['new_cases'].iloc[i]
            s_next_ =  df_final_nonpharma_temp['new_cases'].iloc[i+14]
            if np.abs(s_next_-s_today_)>0:
                df_final_nonpharma_temp['reward_14'].iloc[i] = - np.sign(s_next_-s_today_)
            else:
                df_final_nonpharma_temp['reward_14'].iloc[i] = 1 
    
df_final_nonpharma =  df_final_nonpharma_temp[df_final_nonpharma_temp['Day'] <= '2021-03-31'] # here we constrain the date
df_final_nonpharma['done'] = np.where(df_final_nonpharma['Day'] == '2021-03-31',1,0) 
 

# this includes new cases as rewards and face covering as actions
df_final_nonpharma.to_csv(r'C:/Users/pabgon/rl_representations/data/df_rewards_new_cases_23_March.csv', index = False)


# let s define again the rewards for the nonpharma df using reproduction rate
# reproduction rate has NA
df_final_rl_update_2 =df_final_rl_update.dropna(subset=['reproduction_rate'])
# so let s check how many observation per country
count_per_country = df_final_rl_update_2[['Entity','Day']].groupby('Entity').agg(['count'])
count_per_country[np.array(count_per_country<365)]
'''
>>> count_per_country[np.array(count_per_country<365)]
           Day
         count
Entity
Barbados   329
Belize     305
Bhutan     305
Botswana   336
Burundi    349
Eritrea    357
Fiji        35
Gambia     321
Laos        45
Malawi     362
Suriname   343
Timor      110
'''
df_final_rl_update_2 = filter_rows_by_values(df_final_rl_update_2,'Entity',[
'Fiji', 
'Laos',
'Timor' 
])

df_final_nonpharma_temp = df_final_rl_update_2[df_final_rl_update_2['Day'] <= '2021-04-30'] #with more days than 2021-03-31 in order to have data of the last seven values for each country
df_final_nonpharma_temp['reward_5'] = None
df_final_nonpharma_temp['reward_14'] = None
for i in range(len(df_final_nonpharma_temp)):
    if i+5<len(df_final_nonpharma_temp):        
        if df_final_nonpharma_temp['Entity'].iloc[i] == df_final_nonpharma_temp['Entity'].iloc[i+5]:
                s_today =  df_final_nonpharma_temp['reproduction_rate'].iloc[i]
                s_next_week =  df_final_nonpharma_temp['reproduction_rate'].iloc[i+5]
                if np.abs(s_next_week-s_today)>0:
                    df_final_nonpharma_temp['reward_5'].iloc[i] = - np.sign(s_next_week-s_today)
                else:
                    df_final_nonpharma_temp['reward_5'].iloc[i] = 1
    if i+14<len(df_final_nonpharma_temp): 
        if df_final_nonpharma_temp['Entity'].iloc[i] == df_final_nonpharma_temp['Entity'].iloc[i+14]:
            s_today_ =  df_final_nonpharma_temp['reproduction_rate'].iloc[i]
            s_next_ =  df_final_nonpharma_temp['reproduction_rate'].iloc[i+14]
            if np.abs(s_next_-s_today_)>0:
                df_final_nonpharma_temp['reward_14'].iloc[i] = - np.sign(s_next_-s_today_)
            else:
                df_final_nonpharma_temp['reward_14'].iloc[i] = 1 
    
df_final_nonpharma =  df_final_nonpharma_temp[df_final_nonpharma_temp['Day'] <= '2021-03-31'] # here we constrain the date
df_final_nonpharma['done'] = np.where(df_final_nonpharma['Day'] == '2021-03-31',1,0) 

df_final_nonpharma.to_csv(r'C:/Users/pabgon/rl_representations/data/df_rewards_reproduction_rate_23March2022.csv', index = False)                                                      

