#library('RcppCNPy')
library(reshape2)
library(ggplot2)
#library(rjson)
library(reticulate)

np <- import("numpy")

# Countries

#DQN
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/dqn_reproduction_rate")

countries_1 <- np$load("countries_test_set_rnd_state1.npy",allow_pickle = TRUE)
countries_2 <- np$load("countries_test_set_rnd_state2.npy",allow_pickle = TRUE)
countries_3 <- np$load("countries_test_set_rnd_state3.npy",allow_pickle = TRUE)
countries_4 <- np$load("countries_test_set_rnd_state4.npy",allow_pickle = TRUE)
countries_5 <- np$load("countries_test_set_rnd_state5.npy",allow_pickle = TRUE)
#countries_5 <- np$load("countries_test_set_rnd_state5.npy",allow_pickle = TRUE)
#countries_10 <- np$load("countries_test_set_rnd_state10.npy",allow_pickle = TRUE)
#countries_15 <- np$load("countries_test_set_rnd_state15.npy",allow_pickle = TRUE)
#countries_20 <- np$load("countries_test_set_rnd_state20.npy",allow_pickle = TRUE)

#dBQC


#########   Actions  ###############

#DQN
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/dqn_reproduction_rate/DQN")

ACTIONS_dqn = c()
actions = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)

#DQN 14
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/DQN")

ACTIONS_dqn_14 = c()
actions = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn_14 = c(unlist(actions),ACTIONS_dqn_14)
actions = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn_14 = c(unlist(actions),ACTIONS_dqn_14)
actions = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn_14 = c(unlist(actions),ACTIONS_dqn_14)
actions = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn_14 = c(unlist(actions),ACTIONS_dqn_14)
actions = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn_14 = c(unlist(actions),ACTIONS_dqn_14)
#actions_stack = np$load("action_chosen_rnd_state1_lr0.001.npy",allow_pickle = TRUE) # 40000 / 500 = 80 that multiplied by 28 = 2240

#histograms
hist(ACTIONS_dqn)

# dBCQ
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/dbcq_reproduction_rate/dBCQ")
ACTIONS_dbcq = c()
actions = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)

# dBCQ 14
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/dBCQ")
ACTIONS_dbcq_14 = c()
actions = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq_14 = c(unlist(actions),ACTIONS_dbcq_14)
actions = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq_14 = c(unlist(actions),ACTIONS_dbcq_14)
actions = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq_14 = c(unlist(actions),ACTIONS_dbcq_14)
actions = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq_14 = c(unlist(actions),ACTIONS_dbcq_14)
actions = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq_14 = c(unlist(actions),ACTIONS_dbcq_14)

#histograms
hist(ACTIONS_dbcq)

#gov
#df = read.csv('C:/Users/pabgon/rl_representations/data/df_rewards_new_cases_5_May2022.csv', header=TRUE)
df = read.csv('C:/Users/pabgon/rl_representations/data/df_rewards_reproduction_rate_23March2022.csv', header=TRUE)

state_features = c( 'stringency_index', 'new_cases',
                   'population_density', 'gdp_per_capita', 'diabetes_prevalence',
                   'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
                   'life_expectancy', 'hospital_beds_per_thousand' )
action_gov_test = c()
for(i in 1:length(countries_1)) {
  country= countries_1[i]
  action_gov_test = c(action_gov_test,subset(df,Entity==country)$facial_coverings)
}

hist(action_gov_test)

###
data_actions = data.frame(iterations=seq(1,length(df$facial_coverings)),gov=df$facial_coverings,dbcq5=ACTIONS_dbcq, dqn5= ACTIONS_dqn, dbcq14=ACTIONS_dbcq_14, dqn14= ACTIONS_dqn_14 )
data_actions <- reshape2::melt(data_actions, id.vars = c('iterations'))

data_actions$value = factor(data_actions$value,level=c(0,1,2,3,4),labels = c("No pol","Recom", "Req some", "Req most",'Req all'))

#plot
ggplot(data_actions, aes(x=value, fill=variable,color=variable)) +
  geom_histogram( alpha=0.5, position="dodge",stat = "count")+
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text( size = 15),
    axis.text.y = element_text( size = 15),
    axis.title.x = element_text( size = 15),
    axis.title.y = element_text( size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "Actions", y='Frequency')+ labs(title = "")

## 14 gap

#DQN
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/DQN")

ACTIONS_dqn = c()
actions = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)
actions = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dqn = c(unlist(actions),ACTIONS_dqn)

#actions_stack = np$load("action_chosen_rnd_state1_lr0.001.npy",allow_pickle = TRUE) # 40000 / 500 = 80 that multiplied by 28 = 2240

#histograms
hist(ACTIONS_dqn)

# dBCQ
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/dBCQ")
ACTIONS_dbcq = c()
actions = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)
actions = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)
ACTIONS_dbcq = c(unlist(actions),ACTIONS_dbcq)


###
data_actions = data.frame(iterations=seq(1,length(df$facial_coverings)),gov=df$facial_coverings,dbcq=ACTIONS_dbcq, dqn= ACTIONS_dqn )
data_actions <- reshape2::melt(data_actions, id.vars = c('iterations'))

data_actions$value = factor(data_actions$value,level=c(0,1,2,3,4),labels = c("No pol","Recom", "Req some", "Req most",'Req all'))

#plot
ggplot(data_actions, aes(x=value, fill=variable,color=variable)) +
  geom_histogram( alpha=0.5, position="dodge",stat = "count")+
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text( size = 15),
    axis.text.y = element_text( size = 15),
    axis.title.x = element_text( size = 15),
    axis.title.y = element_text( size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "Actions", y='Frequency')+ labs(title = "")

#### Actions through time


#DQN
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/dqn_reproduction_rate/DQN")

countries_1 <- np$load("countries_test_set_rnd_state1.npy",allow_pickle = TRUE)
countries_2 <- np$load("countries_test_set_rnd_state2.npy",allow_pickle = TRUE)
countries_3 <- np$load("countries_test_set_rnd_state3.npy",allow_pickle = TRUE)
countries_4 <- np$load("countries_test_set_rnd_state4.npy",allow_pickle = TRUE)
countries_5 <- np$load("countries_test_set_rnd_state5.npy",allow_pickle = TRUE)

actions_dqn_1 = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
actions_dqn_2 = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
actions_dqn_3 = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
actions_dqn_4 = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
actions_dqn_5 = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)

# correlation
install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")

chart.Correlation(data_fold1[, c(1,2)], histogram=TRUE,  pch=19)

observed_actions_1 = c()
for(i in 1:length(countries_1)) {
  country= countries_1[i]
  observed_actions_1 = rbind(observed_actions_1,subset(df,Entity==country)['facial_coverings'])
}

state_features = c( 'stringency_index', 'new_cases',
                    'population_density', 'gdp_per_capita', 'diabetes_prevalence',
                    'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
                    'life_expectancy', 'hospital_beds_per_thousand' )
features_data_1 = c()
for(i in 1:length(countries_1)) {
  country= countries_1[i]
  features_data_1 = rbind(features_data_1,subset(df,Entity==country)[state_features])
}

obs_data_fold1=data.frame(action=observed_actions_1,features_data_1)
rl_data_fold1=data.frame(action=unlist(actions_dqn_1),features_data_1)
chart.Correlation(obs_data_fold1[, c(1,2:4)], histogram=FALSE,  pch=19)
chart.Correlation(rl_data_fold1[, c(1,2:4)], histogram=FALSE,  pch=19)

#DQN
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/DQN")

countries_1 <- np$load("countries_test_set_rnd_state1.npy",allow_pickle = TRUE)
countries_2 <- np$load("countries_test_set_rnd_state2.npy",allow_pickle = TRUE)
countries_3 <- np$load("countries_test_set_rnd_state3.npy",allow_pickle = TRUE)
countries_4 <- np$load("countries_test_set_rnd_state4.npy",allow_pickle = TRUE)
countries_5 <- np$load("countries_test_set_rnd_state5.npy",allow_pickle = TRUE)

actions_dqn14_1 = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
actions_dqn14_2 = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
actions_dqn14_3 = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
actions_dqn14_4 = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
actions_dqn14_5 = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)


#dBCQ
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/dbcq_reproduction_rate/dBCQ")

countries_1 <- np$load("countries_test_set_rnd_state1.npy",allow_pickle = TRUE)
countries_2 <- np$load("countries_test_set_rnd_state2.npy",allow_pickle = TRUE)
countries_3 <- np$load("countries_test_set_rnd_state3.npy",allow_pickle = TRUE)
countries_4 <- np$load("countries_test_set_rnd_state4.npy",allow_pickle = TRUE)
countries_5 <- np$load("countries_test_set_rnd_state5.npy",allow_pickle = TRUE)

actions_dbcq_1 = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq_2 = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq_3 = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq_4 = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq_5 = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq_5[[2]]

#dBCQ
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/dBCQ")

countries_1 <- np$load("countries_test_set_rnd_state1.npy",allow_pickle = TRUE)
countries_2 <- np$load("countries_test_set_rnd_state2.npy",allow_pickle = TRUE)
countries_3 <- np$load("countries_test_set_rnd_state3.npy",allow_pickle = TRUE)
countries_4 <- np$load("countries_test_set_rnd_state4.npy",allow_pickle = TRUE)
countries_5 <- np$load("countries_test_set_rnd_state5.npy",allow_pickle = TRUE)

actions_dbcq14_1 = np$load("action_chosen_rnd_state1iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq14_2 = np$load("action_chosen_rnd_state2iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq14_3 = np$load("action_chosen_rnd_state3iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq14_4 = np$load("action_chosen_rnd_state4iteration_39500.npy",allow_pickle = TRUE)
actions_dbcq14_5 = np$load("action_chosen_rnd_state5iteration_39500.npy",allow_pickle = TRUE)

# Australia
df_australia = df[df['Entity']=='Australia',]
df_australia$Day = as.Date(df_australia$Day, format="%Y-%m-%d")

australia = data.frame(day = df_australia$Day, gov = df_australia$facial_coverings, dbcq5 = actions_dbcq_5[[2]],dbcq14 = actions_dbcq14_5[[2]],  dqn5 = actions_dqn_5[[2]],dqn14 = actions_dqn14_5[[2]])
australia <- reshape2::melt(australia, id.vars = c('day'))
names(australia) <- c('Days','criteria', 'Actions')

ggplot(australia, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(australia, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))


#Sweden
countries_1[22]
df_sweden = df[df['Entity']=='Sweden',]
df_sweden$Day = as.Date(df_sweden$Day, format="%Y-%m-%d")


sweden = data.frame(day = df_sweden$Day, gov = df_sweden$facial_coverings, dbcq5 = actions_dbcq_1[[22]],dbcq14 = actions_dbcq14_1[[22]],  dqn5 = actions_dqn_1[[22]],dqn14 = actions_dqn14_1[[22]] )
sweden <- reshape2::melt(sweden, id.vars = c('day'))
names(sweden) <- c('Days','criteria', 'Actions')

ggplot(sweden, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(sweden, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

# Canada
countries_1[4]
df_canada = df[df['Entity']=='Canada',]
df_canada$Day = as.Date(df_canada$Day, format="%Y-%m-%d")

canada = data.frame(day = df_canada$Day, gov = df_canada$facial_coverings, dbcq5 = actions_dbcq_1[[4]],dbcq14 = actions_dbcq14_1[[4]],  dqn5 = actions_dqn_1[[4]],dqn14 = actions_dqn14_1[[4]])
canada <- reshape2::melt(canada, id.vars = c('day'))
names(canada) <- c('Days','criteria', 'Actions')

ggplot(canada, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')


ggplot(canada, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

#Mexico
countries_2[16]
df_mexico = df[df['Entity']=='Mexico',]
df_mexico$Day = as.Date(df_mexico$Day, format="%Y-%m-%d")

mexico = data.frame(day = df_mexico$Day, gov = df_mexico$facial_coverings, dbcq5 = actions_dbcq_2[[16]],dbcq14 = actions_dbcq14_2[[16]],  dqn5 = actions_dqn_2[[16]],dqn14 = actions_dqn14_2[[16]])
mexico <- reshape2::melt(mexico, id.vars = c('day'))
names(mexico) <- c('Days','criteria', 'Actions')

ggplot(mexico, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')


ggplot(mexico, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))


#Argentina
countries_3[2]
df_argentina = df[df['Entity']=='Argentina',]
df_argentina$Day = as.Date(df_argentina$Day, format="%Y-%m-%d")

argentina = data.frame(day = df_argentina$Day, gov = df_argentina$facial_coverings, dbcq5 = actions_dbcq_3[[2]],dbcq14 = actions_dbcq14_3[[2]],  dqn5 = actions_dqn_3[[2]],dqn14 = actions_dqn14_3[[2]])
argentina <- reshape2::melt(argentina, id.vars = c('day'))
names(argentina) <- c('Days','criteria', 'Actions')

ggplot(argentina, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(argentina, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

# Brazil
countries_4[[4]]
df_brazil = df[df['Entity']=='Brazil',]
df_brazil$Day = as.Date(df_brazil$Day, format="%Y-%m-%d")

brazil = data.frame(day = df_brazil$Day, gov = df_brazil$facial_coverings, dbcq5 = actions_dbcq_4[[4]],dbcq14 = actions_dbcq14_4[[4]],  dqn5 = actions_dqn_4[[4]],dqn14 = actions_dqn14_4[[4]])
brazil <- reshape2::melt(brazil, id.vars = c('day'))
names(brazil) <- c('Days','criteria', 'Actions')

ggplot(brazil, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(brazil, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

#Italy
countries_5[[14]]
df_italy = df[df['Entity']=='Italy',]
df_italy$Day = as.Date(df_italy$Day, format="%Y-%m-%d")

italy = data.frame(day = df_italy$Day, gov = df_italy$facial_coverings, dbcq5 = actions_dbcq_5[[14]],dbcq14 = actions_dbcq14_5[[14]],  dqn5 = actions_dqn_5[[14]],dqn14 = actions_dqn14_5[[14]])
italy <- reshape2::melt(italy, id.vars = c('day'))
names(italy) <- c('Days','criteria', 'Actions')

ggplot(italy, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(italy, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

# Spain
countries_4[22]
df_spain = df[df['Entity']=='Spain',]
df_spain$Day = as.Date(df_spain$Day, format="%Y-%m-%d")

spain = data.frame(day = df_spain$Day, gov = df_spain$facial_coverings, dbcq5 = actions_dbcq_4[[22]],dbcq14 = actions_dbcq14_4[[22]],  dqn5 = actions_dqn_4[[22]],dqn14 = actions_dqn14_4[[22]])
spain <- reshape2::melt(spain, id.vars = c('day'))
names(spain) <- c('Days','criteria', 'Actions')

ggplot(spain, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(spain, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))


#Israel
countries_4[16]
df_israel = df[df['Entity']=='Israel',]
df_israel$Day = as.Date(df_israel$Day, format="%Y-%m-%d")

israel = data.frame(day = df_israel$Day, gov = df_israel$facial_coverings, dbcq5 = actions_dbcq_4[[16]],dbcq14 = actions_dbcq14_4[[16]],  dqn5 = actions_dqn_4[[16]],dqn14 = actions_dqn14_4[[16]])
israel <- reshape2::melt(israel, id.vars = c('day'))
names(israel) <- c('Days','criteria', 'Actions')

ggplot(israel, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(israel, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

# Japon
countries_3[15]
df_japon = df[df['Entity']=='Japan',]
df_japon$Day = as.Date(df_japon$Day, format="%Y-%m-%d")

japon = data.frame(day = df_japon$Day, gov = df_japon$facial_coverings, dbcq5 = actions_dbcq_3[[15]],dbcq14 = actions_dbcq14_3[[15]],  dqn5 = actions_dqn_3[[15]],dqn14 = actions_dqn14_3[[15]])
japon <- reshape2::melt(japon, id.vars = c('day'))
names(japon) <- c('Days','criteria', 'Actions')

ggplot(japon, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

  ggplot(japon, aes(x=Days, y=Actions,color=criteria)) +
    facet_grid(criteria ~ .) +
    geom_point(size = 3) +
    scale_y_continuous(breaks=c(0,1, 2, 3,4))+
    theme(panel.grid.major = element_blank() )+
    theme(panel.grid.major = element_blank(),
          axis.title.x = element_text( size=20),
          axis.title.y = element_text( size=20),
          axis.text.x = element_text( 
            size=20),
          axis.text.y = element_text(size=20)) +
    theme(strip.text.x = element_text(size = 20)) + 
    theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
    labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
    scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))
 

# China
countries_4[5]
df_china = df[df['Entity']=='China',]
df_china$Day = as.Date(df_china$Day, format="%Y-%m-%d")

china = data.frame(day = df_china$Day, gov = df_china$facial_coverings, dbcq5 = actions_dbcq_4[[5]],dbcq14 = actions_dbcq14_4[[5]],  dqn5 = actions_dqn_4[[5]],dqn14 = actions_dqn14_4[[5]])
china <- reshape2::melt(china, id.vars = c('day'))
names(china) <- c('Days','criteria', 'Actions')

ggplot(china, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(china, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

#New Zeland
countries_5[19]
df_newzealand = df[df['Entity']=='New Zealand',]
df_newzealand$Day = as.Date(df_newzealand$Day, format="%Y-%m-%d")

newzealand = data.frame(day = df_newzealand$Day, gov = df_newzealand$facial_coverings, dbcq5 = actions_dbcq_5[[19]],dbcq14 = actions_dbcq14_5[[19]],  dqn5 = actions_dqn_5[[19]],dqn14 = actions_dqn14_5[[19]])
newzealand <- reshape2::melt(newzealand, id.vars = c('day'))
names(newzealand) <- c('Days','criteria', 'Actions')

ggplot(newzealand, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(newzealand, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

# Niger
countries_1[16]
df_niger = df[df['Entity']=='Niger',]
df_niger$Day = as.Date(df_niger$Day, format="%Y-%m-%d")

niger = data.frame(day = df_niger$Day, gov = df_niger$facial_coverings, dbcq5 = actions_dbcq_1[[16]],dbcq14 = actions_dbcq14_1[[16]],  dqn5 = actions_dqn_1[[16]],dqn14 = actions_dqn14_1[[16]])
niger <- reshape2::melt(niger, id.vars = c('day'))
names(niger) <- c('Days','criteria', 'Actions')

ggplot(niger, aes(x=Days, y=Actions,color=criteria)) + 
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')

ggplot(niger, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

#South Africa
countries_5[27]
df_southafrica = df[df['Entity']=='South Africa',]
df_southafrica$Day = as.Date(df_southafrica$Day, format="%Y-%m-%d")

southafrica = data.frame(day = df_southafrica$Day, gov = df_southafrica$facial_coverings, dbcq5 = actions_dbcq_5[[27]],dbcq14 = actions_dbcq14_5[[27]],  dqn5 = actions_dqn_5[[27]],dqn14 = actions_dqn14_5[[27]])
southafrica <- reshape2::melt(southafrica, id.vars = c('day'))
names(southafrica) <- c('Days','criteria', 'Actions')

ggplot(southafrica, aes(x=Days, y=Actions,color=criteria)) +
 facet_grid(criteria ~ .) +
  geom_point(size = 3)+ geom_line()+labs(colour = NULL) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions')


ggplot(southafrica, aes(x=Days, y=Actions,color=criteria)) +
  facet_grid(criteria ~ .) +
  geom_point(size = 3) +
  scale_y_continuous(breaks=c(0,1, 2, 3,4))+
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  labs(title = "") + labs(x = "", y='Actions') + theme(legend.position = "none")+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0,4, 1), labels = c('No pol','Recom','Req some','Req most','Req all'))

########### Q values  #############

Q_values_all_iter = np$load("Q_values_rnd_state1_lr0.001.npy",allow_pickle = TRUE)

for (i in seq(0,39500,500)) {
  file_q <- paste0("Q_values_rnd_state1iteration_", i, ".npy")
   q = np$load(file_q,allow_pickle = TRUE)
  for (j in 1:dim(q)){
    q_temp = q[[j]]
    
  }
  
}

aa= np$load("Q_values_rnd_state1iteration_500.npy",allow_pickle = TRUE)
for (j in 1:dim(aa)){
  q_temp = aa[[j]]
  
}

####### Evaluation 

##DQN
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/DQN")

eval_dr = np$load("DQN_policy_eval_dr_rnd_state3_lr0.001.npy",allow_pickle = TRUE)

#mean(eval_dr[76:80,1])
#mean(eval_dr[76:80,2])

#eval_wis = np$load("DQN_policy_eval_wis_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
#mean(eval_wis[76:80,1])

eval_df = data.frame(iterations=seq(0,40000-1,by=500),DR_rl=eval_dr[,1], DR_gov=eval_dr[,2], WIS_rl= eval_wis[,1], WIS_gov= eval_wis[,2] )
eval_df <- reshape2::melt(eval_df, id.vars = c('iterations'))
names(eval_df) <- c('Iterations','criteria', 'OPE_value')

ggplot(eval_df, aes(x=Iterations, y=OPE_value, color=criteria)) + 
  geom_line(aes(linetype=criteria)) +
  geom_point()   +
  geom_smooth(aes(x = Iterations, y = OPE_value)) +
  labs(y='OPE value') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15) +  theme(legend.position = "bottom") + theme(legend.text = element_text(size = 10))+
  guides(colour=guide_legend(nrow=2,byrow=TRUE))

eval_wis_1 = np$load("DQN_policy_eval_wis_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
eval_wis_2 = np$load("DQN_policy_eval_wis_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
eval_wis_3 = np$load("DQN_policy_eval_wis_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
eval_wis_4 = np$load("DQN_policy_eval_wis_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
eval_wis_5 = np$load("DQN_policy_eval_wis_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

boxplot_df_dqn = data.frame(rl=cbind(c(eval_wis_1[75:80,1],eval_wis_2[75:80,1],eval_wis_3[75:80,1],eval_wis_4[75:80,1],eval_wis_5[75:80,1])),
           gov=cbind(c(eval_wis_1[75:80,2],eval_wis_2[75:80,2],eval_wis_3[75:80,2],eval_wis_4[75:80,2],eval_wis_5[75:80,2])),
           fold = seq(1,30,1))

boxplot_df <- reshape2::melt(boxplot_df, id.vars = c('fold'))

ggplot(boxplot_df, aes(x=variable, y=value, color=variable)) +
  geom_boxplot()+
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text( size = 15),
    axis.text.y = element_text( size = 15),
    axis.title.x = element_text( size = 15),
    axis.title.y = element_text( size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "", y='WIS Return')+ labs(title = "DQN")

boxplot_df = data.frame(rl=rbind(eval_wis_1[80,1],eval_wis_2[80,1],eval_wis_3[80,1],eval_wis_4[80,1],eval_wis_5[80,1]),
                        gov=rbind(eval_wis_1[80,2],eval_wis_2[80,2],eval_wis_3[80,2],eval_wis_4[80,2],eval_wis_5[80,2])
                        )

boxplot_df <- reshape2::melt(boxplot_df, id.vars = c('fold'))

ggplot(boxplot_df, aes(x=variable, y=value, color=variable)) +
  geom_boxplot()


### dBCQ
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/dBCQ")

eval_dr = np$load("dBCQ_policy_eval_dr_rnd_state4_lr0.001.npy",allow_pickle = TRUE)


eval_wis_1 = np$load("dBCQ_policy_eval_wis_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
eval_wis_2 = np$load("dBCQ_policy_eval_wis_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
eval_wis_3 = np$load("dBCQ_policy_eval_wis_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
eval_wis_4 = np$load("dBCQ_policy_eval_wis_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
eval_wis_5 = np$load("dBCQ_policy_eval_wis_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

boxplot_df_dbcq = data.frame(rl=cbind(c(eval_wis_1[75:80,1],eval_wis_2[75:80,1],eval_wis_3[75:80,1],eval_wis_4[75:80,1],eval_wis_5[75:80,1])),
                        gov=cbind(c(eval_wis_1[75:80,2],eval_wis_2[75:80,2],eval_wis_3[75:80,2],eval_wis_4[75:80,2],eval_wis_5[75:80,2])),
                        fold = seq(1,30,1))

boxplot_df <- reshape2::melt(boxplot_df, id.vars = c('fold'))

ggplot(boxplot_df, aes(x=variable, y=value, color=variable)) +
  geom_boxplot()+
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text( size = 15),
    axis.text.y = element_text( size = 15),
    axis.title.x = element_text( size = 15),
    axis.title.y = element_text( size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "", y='WIS Return') + ggtitle("dBCQ") 
#+labs(title = "dBCQ")

# dqn lstm
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/DQN_lstm")

eval_dr = np$load("DQN_policy_eval_dr_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

#mean(eval_dr[76:80,1])
#mean(eval_dr[76:80,2])

#eval_wis = np$load("DQN_policy_eval_wis_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
#mean(eval_wis[76:80,1])

eval_df = data.frame(iterations=seq(0,40000-1,by=500),DR_rl=eval_dr[,1], DR_gov=eval_dr[,2], WIS_rl= eval_wis[,1], WIS_gov= eval_wis[,2] )
eval_df <- reshape2::melt(eval_df, id.vars = c('iterations'))
names(eval_df) <- c('Iterations','criteria', 'OPE_value')

ggplot(eval_df, aes(x=Iterations, y=OPE_value, color=criteria)) + 
  geom_line(aes(linetype=criteria)) +
  geom_point()   +
  geom_smooth(aes(x = Iterations, y = OPE_value)) +
  labs(y='OPE value') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=15),
        axis.title.y = element_text( size=15),
        axis.text.x = element_text( 
          size=15),
        axis.text.y = element_text(size=15)) +
  theme(strip.text.x = element_text(size = 15)) + 
  theme(strip.text.y = element_text(size = 15)) + theme_bw(base_size = 15) +  theme(legend.position = "bottom") + theme(legend.text = element_text(size = 10))+
  guides(colour=guide_legend(nrow=2,byrow=TRUE))

eval_wis_1 = np$load("DQN_policy_eval_wis_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
eval_wis_2 = np$load("DQN_policy_eval_wis_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
eval_wis_3 = np$load("DQN_policy_eval_wis_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
eval_wis_4 = np$load("DQN_policy_eval_wis_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
eval_wis_5 = np$load("DQN_policy_eval_wis_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

boxplot_df_dqn_lstm = data.frame(rl=cbind(c(eval_wis_1[75:80,1],eval_wis_2[75:80,1],eval_wis_3[75:80,1],eval_wis_4[75:80,1],eval_wis_5[75:80,1])),
                             gov=cbind(c(eval_wis_1[75:80,2],eval_wis_2[75:80,2],eval_wis_3[75:80,2],eval_wis_4[75:80,2],eval_wis_5[75:80,2])),
                             fold = seq(1,30,1))

### All methods together
evaluation_df =data.frame(fold=boxplot_df_dqn[,3],gov=boxplot_df_dqn[,2],dqn=boxplot_df_dqn[,1],dbcq=boxplot_df_dbcq[,1],dqn_lstm = boxplot_df_dqn_lstm[,1])
evaluation_df <- reshape2::melt(evaluation_df, id.vars = c('fold'))

ggplot(evaluation_df, aes(x=variable, y=value, color=variable)) +
  geom_boxplot()+
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text( size = 15),
    axis.text.y = element_text( size = 15),
    axis.title.x = element_text( size = 15),
    axis.title.y = element_text( size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "", y='WIS Return')  


#########  Plot train loss against number of epoch

##DQN
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/DQN")

ggplot(data_train_loss_1,aes(x = iterations, y = loss)) + geom_point() +
  geom_smooth(aes(x = iterations, y = loss)) + 
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15)
  )



ggplot(data_train_loss_2,aes(x = iterations, y = loss)) + geom_point() +
  geom_smooth(aes(x = iterations, y = loss)) + 
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15)
  )


# All train plot smooth curve
loss_train_1 = np$load("loss_train_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
loss_train_2 = np$load("loss_train_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
loss_train_3 = np$load("loss_train_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
loss_train_4 = np$load("loss_train_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
loss_train_5 = np$load("loss_train_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

df_loss_plot= data.frame(iterations=seq(1,40000),fold_1 = loss_train_1, fold_2 = loss_train_2,
                         fold_3 = loss_train_3, fold_4 = loss_train_4, fold_5 = loss_train_5)
df_loss_plot <- reshape2::melt(df_loss_plot, id.vars = c('iterations'))
names(df_loss_plot) <- c('Iterations','Fold', 'Loss')

ggplot(df_loss_plot,aes(x = Iterations, y = Loss, color=Fold)) +
  geom_smooth(aes(x = Iterations, y = Loss)) + 
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "Iterations", y='Loss') + ggtitle("DQN") 
  

# dBCQ
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/dBCQ")
loss_train_1 = np$load("loss_train_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
loss_train_2 = np$load("loss_train_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
loss_train_3 = np$load("loss_train_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
loss_train_4 = np$load("loss_train_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
loss_train_5 = np$load("loss_train_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

df_loss_plot= data.frame(iterations=seq(1,40000),fold_1 = loss_train_1, fold_2 = loss_train_2,
                         fold_3 = loss_train_3, fold_4 = loss_train_4, fold_5 = loss_train_5)
df_loss_plot <- reshape2::melt(df_loss_plot, id.vars = c('iterations'))
names(df_loss_plot) <- c('Iterations','Fold', 'Loss')

ggplot(df_loss_plot,aes(x = Iterations, y = Loss, color=Fold)) +
  geom_smooth(aes(x = Iterations, y = Loss)) + 
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "Iterations", y='Loss') + ggtitle("dBCQ") 

# dBCQ 14
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/dBCQ")
loss_train_1 = np$load("loss_train_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
loss_train_2 = np$load("loss_train_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
loss_train_3 = np$load("loss_train_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
loss_train_4 = np$load("loss_train_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
loss_train_5 = np$load("loss_train_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

df_loss_plot= data.frame(iterations=seq(1,40000),fold_1 = loss_train_1, fold_2 = loss_train_2,
                         fold_3 = loss_train_3, fold_4 = loss_train_4, fold_5 = loss_train_5)
df_loss_plot <- reshape2::melt(df_loss_plot, id.vars = c('iterations'))
names(df_loss_plot) <- c('Iterations','Fold', 'Loss')

ggplot(df_loss_plot,aes(x = Iterations, y = Loss, color=Fold)) +
  geom_smooth(aes(x = Iterations, y = Loss)) + 
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "Iterations", y='Loss') + ggtitle("dBCQ") 

# DQN 14
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap14/states_t/DQN")
loss_train_1 = np$load("loss_train_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
loss_train_2 = np$load("loss_train_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
loss_train_3 = np$load("loss_train_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
loss_train_4 = np$load("loss_train_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
loss_train_5 = np$load("loss_train_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

df_loss_plot= data.frame(iterations=seq(1,40000),fold_1 = loss_train_1, fold_2 = loss_train_2,
                         fold_3 = loss_train_3, fold_4 = loss_train_4, fold_5 = loss_train_5)
df_loss_plot <- reshape2::melt(df_loss_plot, id.vars = c('iterations'))
names(df_loss_plot) <- c('Iterations','Fold', 'Loss')

ggplot(df_loss_plot,aes(x = Iterations, y = Loss, color=Fold)) +
  geom_smooth(aes(x = Iterations, y = Loss)) + 
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "Iterations", y='Loss') + ggtitle("DQN") 



# ´dqn lstm
setwd( "C:/Users/pabgon/rl_representations/srl_test/results/rewards_gap5/states_t/DQN_lstm")
loss_train_1 = np$load("loss_train_rnd_state1_lr0.001.npy",allow_pickle = TRUE)
loss_train_2 = np$load("loss_train_rnd_state2_lr0.001.npy",allow_pickle = TRUE)
loss_train_3 = np$load("loss_train_rnd_state3_lr0.001.npy",allow_pickle = TRUE)
loss_train_4 = np$load("loss_train_rnd_state4_lr0.001.npy",allow_pickle = TRUE)
loss_train_5 = np$load("loss_train_rnd_state5_lr0.001.npy",allow_pickle = TRUE)

df_loss_plot= data.frame(iterations=seq(1,40000),fold_1 = loss_train_1, fold_2 = loss_train_2,
                         fold_3 = loss_train_3, fold_4 = loss_train_4, fold_5 = loss_train_5)
df_loss_plot <- reshape2::melt(df_loss_plot, id.vars = c('iterations'))
names(df_loss_plot) <- c('Iterations','Fold', 'Loss')

ggplot(df_loss_plot,aes(x = Iterations, y = Loss, color=Fold)) +
  geom_smooth(aes(x = Iterations, y = Loss)) + 
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(face = "bold", size = 12),
    axis.text.y = element_text(face = "bold", size = 12),
    axis.title.x = element_text(face = "bold", size = 15),
    axis.title.y = element_text(face = "bold", size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "Iterations", y='Loss') + ggtitle("DQN_LSTM") 

######################################################################

dqn5_wis=c(173.1882967038509,
-0.3067040491799888,
132.43481150312837,
-1.6473223458652924,
83.34528605769397)

dbcq5_wis = c(0.1626931782359492,
-21.084885555555715,
-40.27043215013228,
38.52991159936234,
88.02481012103027)

dbcq14_wis = c(-17.79318491092939, 20.17637318213469, 52.428929718491204, 12.73734218243196, 94.21199821943024)

dqn14_wis = c(30.310148964443087, 71.456181350023, 126.44890154268951, 1.1910059865275913, 236.36830408929174)


gov_wis =c(25.35682637716505,
22.172711626538934,
20.690873652610954,
28.05595775318912,
26.08476324392621)

evaluation_df =data.frame(fold=seq(1,5),gov=gov_wis,dqn5=dqn5_wis,dbcq5=dbcq5_wis,dqn14=dqn14_wis,dbcq14=dbcq14_wis)
evaluation_df <- reshape2::melt(evaluation_df, id.vars = c('fold'))

ggplot(evaluation_df, aes(x=variable, y=value, color=variable)) +
  geom_boxplot()+
  theme_bw() +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 15),
    axis.text.x = element_text( size = 15),
    axis.text.y = element_text( size = 15),
    axis.title.x = element_text( size = 15),
    axis.title.y = element_text( size = 15),
    plot.title = element_text(hjust = 0.5,size = 15)
  )+ labs(x = "", y='WIS Return')  