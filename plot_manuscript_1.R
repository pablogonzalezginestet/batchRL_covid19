

library(tableone)


df = read.csv('C:/Users/pabgon/rl_representations/data/df_rewards_new_cases_5_May2022.csv', header=TRUE)

state_features = c( 'stringency_index', 'new_cases',
                    'population_density', 'gdp_per_capita', 'diabetes_prevalence',
                    'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
                    'life_expectancy', 'hospital_beds_per_thousand' )

myVars <-  state_features

df = data.frame(df)
CreateTableOne(vars = myVars, data = df)

CreateTableOne(data = df)

summary(df)


### country list
library(xtable)
df$day1 <- as.Date(df$Day)
df[df$day1 == min(date_df),]

df

result <- table(unique(df['Entity']))
xtable(result)

print(xtable(unique(df['Entity'])), include.rownames=T, include.colnames=T)


library(plyr)

ddd = ddply(df[c('Entity','Day')], .(Entity), function(x) x[c(1, nrow(x)), ])

library(tidyr)

aaa = dcast(setDT(ddd), Entity~rowid(Entity), value.var = 'Day')

aaa_by_date=aaa[order(aaa$`1`),]

print(xtable(aaa[1:28,]), include.rownames=F, include.colnames=T)
print(xtable(aaa[29:56,]), include.rownames=F, include.colnames=T)
print(xtable(aaa[57:84,]), include.rownames=F, include.colnames=T)
print(xtable(aaa[85:112,]), include.rownames=F, include.colnames=T)
print(xtable(aaa[113:140,]), include.rownames=F, include.colnames=T)

print(xtable(aaa_by_date[1:28,]), include.rownames=F, include.colnames=T)
print(xtable(aaa_by_date[29:56,]), include.rownames=F, include.colnames=T)
print(xtable(aaa_by_date[57:84,]), include.rownames=F, include.colnames=T)
print(xtable(aaa_by_date[85:112,]), include.rownames=F, include.colnames=T)
print(xtable(aaa_by_date[113:140,]), include.rownames=F, include.colnames=T)

####### correlation between actions and features
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


ACTIONS_1 = unlist(actions_dqn_1)
state_features = c( 'stringency_index', 'new_cases',
                    'population_density', 'gdp_per_capita', 'diabetes_prevalence',
                    'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
                    'life_expectancy', 'hospital_beds_per_thousand' )
features_data_1 = c()
for(i in 1:length(countries_1)) {
  country= countries_1[i]
  features_data_1 = rbind(features_data_1,subset(df,Entity==country)[state_features])
}

table(ACTIONS_1, features_data_1$hospital_beds_per_thousand)

data_fold1=data.frame(action=ACTIONS_1,features_data_1)

install.packages("PerformanceAnalytics")
library("PerformanceAnalytics")

my_data <- mtcars[, c(1,3,4,5,6,7)]
chart.Correlation(data_fold1[, c(1,2)], histogram=TRUE,  pch=19)

observed_actions_1 = c()
for(i in 1:length(countries_1)) {
  country= countries_1[i]
  observed_actions_1 = rbind(observed_actions_1,subset(df,Entity==country)['facial_coverings'])
}

obs_data_fold1=data.frame(action=observed_actions_1,features_data_1)
chart.Correlation(obs_data_fold1[, c(1,2)], histogram=TRUE,  pch=19)
chart.Correlation(obs_data_fold1[, c(1,3)], histogram=TRUE,  pch=19)

# multinomial regression
rl_actions_reg = multinom(action ~  stringency_index + new_cases + population_density + gdp_per_capita + diabetes_prevalence +
           cardiovasc_death_rate + aged_65_older + human_development_index + life_expectancy + hospital_beds_per_thousand , data = data_fold1)

summary(rl_actions_reg)
exp(coef(rl_actions_reg))

obs_actions_reg = multinom(facial_coverings ~  stringency_index + new_cases + population_density + gdp_per_capita + diabetes_prevalence +
                            cardiovasc_death_rate + aged_65_older + human_development_index + life_expectancy + hospital_beds_per_thousand  , data = obs_data_fold1)

summary(obs_actions_reg)
exp(coef(obs_actions_reg))


#### plot face coverings, stringency index and new cases

df = read.csv('C:/Users/pabgon/rl_representations/data/df_rewards_reproduction_rate_23March2022.csv', header=TRUE) 

state_features = c( 'stringency_index', 'new_cases',
                    'population_density', 'gdp_per_capita', 'diabetes_prevalence',
                    'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
                    'life_expectancy', 'hospital_beds_per_thousand' )

df$day1 <- as.Date(df$Day)
df['stringency_index']
library(data.table)
DT <- data.table(df)

bbb = DT[, mean(stringency_index), by = day1]

aaa = DT[, mean(facial_coverings), by = day1]

aaa1 = DT[, mean(reproduction_rate), by = day1]

aaa2 = DT[, mean(reward_5), by = day1]

aaa3 = DT[, mean(reward_14), by = day1]

data_plot = data.frame(day= bbb$day1, stringency_index=bbb$V1, facial_coverings=aaa$V1, R=aaa1$V1,reward=aaa2$V1,reward14=aaa3$V1)
# Start with a usual ggplot2 call:


ggplot(data_plot, aes(day, facial_coverings)) +
  geom_line() +
  geom_smooth(aes(x = day, y = facial_coverings))+
  labs(y='Facial Covering') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +  theme(legend.position = "bottom") + theme(legend.text = element_text(size = 10))+
  guides(colour=guide_legend(nrow=2,byrow=TRUE))+
  scale_y_continuous(limits = c(0, 4), breaks = seq(0, 4, 1))


ggplot(data_plot, aes(day,stringency_index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = stringency_index))+
  labs(y='Stringency index') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) 


ggplot(data_plot, aes(day,R)) +
  geom_line() +
  geom_smooth(aes(x = day, y = R))+
  labs(y='Reproduction rate of COVID-19 ') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
scale_y_continuous(limits = c(0, 3.5), breaks = seq(0, 3.5, .5))


ggplot(data_plot, aes(day,reward)) +
  geom_line() +
  geom_smooth(aes(x = day, y = reward))+
  labs(y='Reward, k=5') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20) +
  scale_y_continuous(limits = c(-1, 1), breaks = seq(-1,1, .5))

ggplot(data_plot, aes(day,reward14)) +
  geom_line() +
  geom_smooth(aes(x = day, y = reward14))+
  labs(y='Reward, k=14') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( 
          size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)+
  scale_y_continuous(limits = c(-1, 1), breaks = seq(-1,1, .5))


data_plot[data_plot$new_cases==max(data_plot$new_cases),]


##### Rewards
df[df['Entity']=='Canada',]$reward_5
df[df['Entity']=='Canada',]$new_cases

 <- reshape2::melt(, id.vars = c('day'))
names() <- c('Days','criteria', 'Actions')


###
df_rw = read.csv('C:/Users/pabgon/rl_representations/data/df_rewards_new_cases_5_May2022.csv', header=TRUE)
df_rep = read.csv('C:/Users/pabgon/rl_representations/data/df_rewards_reproduction_rate_23March2022.csv', header=TRUE) 

state_features = c( 'stringency_index', 'new_cases',
                    'population_density', 'gdp_per_capita', 'diabetes_prevalence',
                    'cardiovasc_death_rate', 'aged_65_older', 'human_development_index',
                    'life_expectancy', 'hospital_beds_per_thousand' )

canada = df_rw[df_rw['Entity']=='Canada',]
canada$index <- 1:nrow(canada)  # create index variable
  # retail 80rows for better graphical understanding
loessMod10 <- loess(new_cases ~ index, data=canada, span=1)
smoothed10 <- predict(loessMod10) 
plot(canada$new_cases, x=canada$index, type="l", main="Loess Smoothing", xlab="Date", ylab="New cases")
lines(smoothed10, x=canada$index, col="red")
lines(a, x=canada$index[2:431], col="green")
a=c()
for (i in 2:length(smoothed10)){
  a=c(a,smoothed10[i]-smoothed10[i-1])
  
}
length(a)
rewards_canada = ifelse(a>=0,-1,1)
plot(rewards_canada, x=canada$index[2:431], type="l", main="rewards", xlab="Date", ylab="New cases")
plot(canada$reward_5, x=canada$index, type="l", main="rewards", xlab="Date", ylab="New cases")
# reproduction rate
canada_rep = df_rep[df_rep['Entity']=='Canada',]
canada_rep$index <- 1:nrow(canada_rep) 
loessMod10 <- loess(reproduction_rate ~ index, data=canada_rep, span=0.50)
smoothed10 <- predict(loessMod10) 
plot(canada_rep$reproduction_rate, x=canada_rep$index, type="l", main="Loess Smoothing and Prediction", xlab="Date", ylab="New cases")
plot(canada_rep$reward_5, x=canada_rep$index, type="l", main="Loess Smoothing and Prediction", xlab="Date", ylab="New cases")
lines(smoothed10, x=canada_rep$index, col="green")


# stringency per country
# oceania
df_oceania1 = data.frame(day = as.Date(df[df['Entity']=='Australia',]$Day) ,index = df[df['Entity']=='Australia',]$stringency_index)

ggplot(df_oceania1, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))


df_oceania2 = data.frame(day = as.Date(df[df['Entity']=='New Zealand',]$Day) ,index = df[df['Entity']=='New Zealand',]$stringency_index)

ggplot(df_oceania2, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

# south america
df_sa1 = data.frame(day = as.Date(df[df['Entity']=='Argentina',]$Day) ,index = df[df['Entity']=='Argentina',]$stringency_index)

ggplot(df_sa1, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

df_sa2 = data.frame(day = as.Date(df[df['Entity']=='Brazil',]$Day) ,index = df[df['Entity']=='Brazil',]$stringency_index)

ggplot(df_sa2, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

# north america

df_na1 = data.frame(day = as.Date(df[df['Entity']=='Canada',]$Day) ,index = df[df['Entity']=='Canada',]$stringency_index)

ggplot(df_na1, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

df_na2 = data.frame(day = as.Date(df[df['Entity']=='Mexico',]$Day) ,index = df[df['Entity']=='Mexico',]$stringency_index)

ggplot(df_na2, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

#asia
df_asia1 = data.frame(day = as.Date(df[df['Entity']=='China',]$Day) ,index = df[df['Entity']=='China',]$stringency_index)

ggplot(df_asia1, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

df_asia2 = data.frame(day = as.Date(df[df['Entity']=='Japan',]$Day) ,index = df[df['Entity']=='Japan',]$stringency_index)

ggplot(df_asia2, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

### europe

df_europe1 = data.frame(day = as.Date(df[df['Entity']=='Italy',]$Day) ,index = df[df['Entity']=='Italy',]$stringency_index)

ggplot(df_europe1, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

df_europe2 = data.frame(day = as.Date(df[df['Entity']=='Spain',]$Day) ,index = df[df['Entity']=='Spain',]$stringency_index)

ggplot(df_europe2, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

df_europe3 = data.frame(day = as.Date(df[df['Entity']=='Sweden',]$Day) ,index = df[df['Entity']=='Sweden',]$stringency_index)

ggplot(df_europe3, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

#africa
df_africa1 = data.frame(day = as.Date(df[df['Entity']=='Niger',]$Day) ,index = df[df['Entity']=='Niger',]$stringency_index)

ggplot(df_africa1, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

df_africa2 = data.frame(day = as.Date(df[df['Entity']=='South Africa',]$Day) ,index = df[df['Entity']=='South Africa',]$stringency_index)

ggplot(df_africa2, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))

#israel
df_israel = data.frame(day = as.Date(df[df['Entity']=='Israel',]$Day) ,index = df[df['Entity']=='Israel',]$stringency_index)

ggplot(df_israel, aes(day,index)) +
  geom_line() +
  geom_smooth(aes(x = day, y = index))+
  labs(y='Stringency index', x='') + 
  theme(panel.grid.major = element_blank() )+
  theme(panel.grid.major = element_blank(),
        axis.title.x = element_text( size=20),
        axis.title.y = element_text( size=20),
        axis.text.x = element_text( size=20),
        axis.text.y = element_text(size=20)) +
  theme(strip.text.x = element_text(size = 20)) + 
  theme(strip.text.y = element_text(size = 20)) + theme_bw(base_size = 20)  +
  scale_y_continuous(limits = c(0, 100), breaks = seq(0,100, 20))


