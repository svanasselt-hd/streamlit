#!/usr/bin/env python
# coding: utf-8

# In[69]:


import streamlit as st
st.set_page_config(
    page_title="World Cup Analysis",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [16, 7]

# sns.set( font_scale = 2, style = 'whitegrid',rc = {'figure.figsize':(20,7)})


# # **STEP 1. Reading data**
# 
# 
# 
# * *Let's read our data and look at them*

# In[70]:


df = pd.read_csv('Fifa_world_cup_matches.csv')


# In[71]:


#df.head(5)


# In[72]:


#df.columns


# # **STEP 2. Preparing of data**
# 
# * *Let's choose the columns that we need to our research*
# 
# 
# 

# In[73]:


df_all =  df[['team1', 'team2', 'possession team1', 'possession team2',
       'possession in contest', 'number of goals team1',
       'number of goals team2', 'date', 'hour', 'category',
       'total attempts team1', 'total attempts team2', 'assists team1', 'assists team2',
       'on target attempts team1', 'on target attempts team2',
       'off target attempts team1', 'off target attempts team2',
       'receptions between midfield and defensive lines team1',
       'receptions between midfield and defensive lines team2',
       'yellow cards team1',
       'yellow cards team2', 'red cards team1', 'red cards team2',
       'fouls against team1', 'fouls against team2', 'offsides team1',
       'offsides team2', 'passes team1', 'passes team2',
       'passes completed team1', 'passes completed team2',
       'corners team1', 'corners team2', 'free kicks team1',
       'free kicks team2', 'penalties scored team1', 'penalties scored team2',
       'own goals team1',
       'own goals team2', 'forced turnovers team1', 'forced turnovers team2',
       'defensive pressures applied team1',
       'defensive pressures applied team2']].copy()


# In[74]:


#df_all.dtypes


# * *I think we should replace all spaces from our column titles*
# * *It will help us to avoid some problems in future analysis*

# In[75]:


df_all.columns = df_all.columns.str.replace(" ", "_")
#df_all.columns


# * *Continue preparing. I'd like to change the type of our date column*

# In[76]:


df_all.date = pd.to_datetime(df_all.date)


# In[77]:


#df_all.head()


# * *Also I noticed that if we want to make some aggregations with our information about possessions we should replace a percentage sign and change a columns type to numeric*

# In[78]:


df_all.possession_team1 = df_all.possession_team1.apply(lambda x: x.replace("%",''))
df_all.possession_team2 = df_all.possession_team2.apply(lambda x: x.replace("%",''))
df_all.possession_in_contest = df_all.possession_in_contest.apply(lambda x: x.replace("%",''))


# In[79]:


df_all.possession_team1 = pd.to_numeric(df_all.possession_team1)
df_all.possession_team2 = pd.to_numeric(df_all.possession_team2)
df_all.possession_in_contest = pd.to_numeric(df_all.possession_in_contest)


# In[80]:


#df_all.head()


# # STEP 3. Questions
# 
# ***Ok, it seems that our data is ready to analysis. Let's create a questions list for our research***
# 
# Directions for analysis:
# 
# 1. Teams goals and passes. Passes accuracy.
# 2. The relationship between ball possession and game result.
# 3. Offsides, red and yellow card statistic.
# 4. Compare of different team playing styles.
# 5. The accumulated wearness influence on finalist teams.
# 

# ***Teams goals and passes. Passes accuracy***

# In[81]:


home_goals = df_all.groupby('team1') \
    .agg({"number_of_goals_team1":"sum",
          'total_attempts_team1' : "sum",
          'assists_team1' : "sum",
          'on_target_attempts_team1' : "sum",
          'off_target_attempts_team1' : "sum"}) \
    .reset_index()
#home_goals.head()


# In[82]:


away_goals = df_all.groupby('team2') \
    .agg({"number_of_goals_team2":"sum",
          'total_attempts_team2' : "sum",
          'assists_team2' : "sum",
          'on_target_attempts_team2' : "sum",
          'off_target_attempts_team2' : "sum"}) \
    .reset_index()
#away_goals.head()


# In[83]:


total_goals = home_goals.merge(away_goals, left_on = 'team1', right_on = 'team2' )
#total_goals.head()


# In[84]:


total_goals  = total_goals.drop('team2', axis=1)
#total_goals.head()


# In[85]:


total_goals = total_goals \
    .rename(columns={"number_of_goals_team1":"home_goals",
                     "number_of_goals_team2":"away_goals",
                     "total_attempts_team1" : "home_attempts_total",
                     "on_target_attempts_team1" : "home_attempts_on_target",
                     "off_target_attempts_team1" : "home_attempts_off_target",
                     "assists_team1" : "home_assists",
                     "total_attempts_team2" : "away_attempts_total",
                     "on_target_attempts_team2" : "away_attempts_on_target",
                     "off_target_attempts_team2" : "away_attempts_off_target",
                     "assists_team2" : "away_assists",
                     "team1" : "team"
                    })
total_goals = total_goals.assign(total_goals  = total_goals.home_goals + total_goals.away_goals)
total_goals = total_goals.assign(total_assists  = total_goals.home_assists + total_goals.away_assists)
total_goals = total_goals.assign(total_attempts  = total_goals.home_attempts_total + total_goals.away_attempts_total)
total_goals = total_goals.assign(total_attempts_on_target  = total_goals.home_attempts_on_target + total_goals.away_attempts_on_target)
total_goals = total_goals.assign(total_attempts_off_target  = total_goals.home_attempts_off_target + total_goals.away_attempts_off_target)
#total_goals.head()


# In[86]:


#total_goals.columns


# In[87]:


total_goals = total_goals[['team', 'total_goals','home_goals', 'away_goals', 
                           'total_assists', 'home_assists', 'away_assists',
                           'total_attempts', 'home_attempts_total', 'away_attempts_total',
                           'total_attempts_on_target', 'home_attempts_on_target', 'away_attempts_on_target',
                           'total_attempts_off_target', 'home_attempts_off_target', 'away_attempts_off_target']].set_index('team')


# In[88]:


top_7_total_goals = total_goals.sort_values('total_goals',ascending=False).head(7)
top_7_total_goals


# In[89]:


top_7_total_goals.total_goals.plot(kind = 'barh', title = 'TOP 7 most scored team\n')
plt.show()


# In[90]:


# assume that top_7_total_goals is a pandas DataFrame containing the data for the chart
fig, ax = plt.subplots()
top_7_total_goals.total_goals.plot(kind='barh', ax=ax)
ax.set_title('TOP 7 most scored team')
ax.set_xlabel('Total Goals')

# display the plot in the Streamlit app
st.pyplot(fig)


# In[91]:


top_7_total_goals[['total_goals', 'home_goals', 'away_goals']].plot(kind="barh", title = 'Goals distribution\n')
plt.show()


# In[92]:


fig, ax = plt.subplots()
top_7_total_goals[['total_goals', 'home_goals', 'away_goals']].plot(kind='barh', ax=ax)
ax.set_title('Goals distribution')
ax.set_xlabel('Goals')

# display the plot in the Streamlit app
st.pyplot(fig)


# *It's interesting to notice that despite the fact that all matches were in Qatar and teams did not have the advantage of their own field(except Qatar team)  whatever most goals by top-teams were scored when they played their "home" matches.*

# In[93]:


top_7_total_assists = total_goals.sort_values('total_assists',ascending=False).head(7)
top_7_total_assists[['total_assists', 'home_assists', 'away_assists']].plot(kind="barh", title = 'Assists distribution\n')
plt.show()


# In[94]:


# assume that top_7_total_assists is a pandas DataFrame containing the data for the chart
fig, ax = plt.subplots()
top_7_total_assists[['total_assists', 'home_assists', 'away_assists']].plot(kind='barh', ax=ax)
ax.set_title('Assists distribution')
ax.set_xlabel('Assists')

# display the plot in the Streamlit app
st.pyplot(fig)


# *The same story with assists!*

# In[95]:


(total_goals.total_goals / total_goals.total_attempts * 100) \
    .round(2) \
    .sort_values(ascending=False) \
    .head(7) \
    .plot(kind='barh', title = 'TOP 7 teams with highest attempts realisation percentage\n')
plt.show()


# In[96]:


# assume that total_goals is a pandas DataFrame containing the data for the chart
top_7_attempts_percentage = (total_goals.total_goals / total_goals.total_attempts * 100) \
                                .round(2) \
                                .sort_values(ascending=False) \
                                .head(7)

fig, ax = plt.subplots()
top_7_attempts_percentage.plot(kind='barh', ax=ax)
ax.set_title('TOP 7 teams with highest attempts realization percentage')
ax.set_xlabel('Attempts realization percentage')

# display the plot in the Streamlit app
st.pyplot(fig)


# In[97]:


passes = df_all[['team1', 'team2','date', 'passes_team1', 'passes_team2',
       'passes_completed_team1', 'passes_completed_team2']]
passes.head()


# *Let's explore team passes and accuracy*

# In[98]:


home_passes = passes.groupby('team1',as_index=False) \
    .agg({"passes_team1" : "sum",
          "passes_completed_team1" : "sum"}) \
    .rename(columns={"team1":"team",
                    "passes_team1":"home_passes_total",
                    "passes_completed_team1":"home_passes_complete_total"}) \
                      
home_passes.head()


# In[99]:


home_passes = home_passes.assign(home_accuracy = home_passes.home_passes_complete_total / home_passes.home_passes_total * 100).round(2)
home_passes.head()


# In[100]:


away_passes = passes.groupby('team2',as_index=False) \
    .agg({"passes_team2" : "sum",
          "passes_completed_team2" : "sum"}) \
    .rename(columns={"team2":"team",
                    "passes_team2":"away_passes_total",
                    "passes_completed_team2":"away_passes_complete_total"}) \
                  
away_passes.head()


# In[101]:


away_passes = away_passes.assign(away_accuracy = away_passes.away_passes_complete_total / away_passes.away_passes_total * 100).round(2)
away_passes.head()


# In[102]:


total_passes = home_passes.merge(away_passes, on = 'team')
total_passes.head()


# In[103]:


total_passes = total_passes \
    .assign(total_passes  = total_passes.home_passes_total + total_passes.away_passes_total)
total_passes = total_passes \
    .assign(total_complete_passes  = total_passes.home_passes_complete_total + total_passes.away_passes_complete_total)
total_passes = total_passes \
    .assign(total_accuracy  = (total_passes.home_accuracy + total_passes.away_accuracy) / 2)

total_passes.head()


# In[104]:


total_passes.columns


# In[105]:


total_passes = total_passes [['team', 'total_passes', 'home_passes_total', 'away_passes_total',
                              'total_complete_passes', 'home_passes_complete_total', 'away_passes_complete_total',
                              'total_accuracy', 'home_accuracy', 'away_accuracy',]]
total_passes.head()


# In[106]:


total_passes.sort_values(by = 'total_accuracy', ascending=False)[['team', 'total_accuracy']] \
    .set_index('team') \
    .plot(kind='barh', title = 'Passes accuracy\n')
plt.xticks(range(72,96,2))
plt.show()


# In[107]:


# assume that total_passes is a pandas DataFrame containing the data for the chart
pass_accuracy_df = total_passes.sort_values(by='total_accuracy', ascending=False)[['team', 'total_accuracy']] \
    .set_index('team')

fig, ax = plt.subplots()
pass_accuracy_df.plot(kind='barh', ax=ax)
ax.set_title('Passes accuracy')
ax.set_xlabel('Accuracy (%)')

# set x-ticks to range from 72 to 96 with a step of 2
ax.set_xticks(range(72, 96, 2))

# display the plot in the Streamlit app
st.pyplot(fig)


# * We can see that Spain is a leader in the passes accuracy statistic with big difference from second place
# * Also I noticed that semi-finalist Morocco team have only 81% of accuracy*

# ***Red and yellow card statistic***

# In[108]:


home_fouls = df_all.groupby('team1',as_index=False) \
    .agg({"yellow_cards_team1" : "mean",
         "red_cards_team1" : "mean",
         "fouls_against_team1" : "mean"}) \
    .rename(columns={"yellow_cards_team1":"home_yellow",
                     "red_cards_team1":"home_red",
                     "fouls_against_team1":"home_fouls_against",
                     "team1" : "team"}) \
    .round(2)
home_fouls.head()


# In[109]:


away_fouls = df_all.groupby('team2',as_index=False) \
    .agg({"yellow_cards_team2" : "mean",
         "red_cards_team2" : "mean",
         "fouls_against_team2" : "mean"}) \
    .rename(columns={"yellow_cards_team2":"away_yellow",
                     "red_cards_team2":"away_red",
                     "fouls_against_team2":"away_fouls_against",
                     "team2" : "team"}) \
    .round(2)
away_fouls.head()


# In[110]:


fouls_and_cards = home_fouls.merge(away_fouls, on = 'team')
fouls_and_cards.head()


# In[111]:


fouls_and_cards = fouls_and_cards.assign(yellow_cards_per_game = (fouls_and_cards.home_yellow + fouls_and_cards.away_yellow) / 2 )
fouls_and_cards = fouls_and_cards.assign(red_cards_per_game = (fouls_and_cards.home_red + fouls_and_cards.away_red) / 2 )
fouls_and_cards = fouls_and_cards.assign(fouls_per_game = (fouls_and_cards.home_fouls_against + fouls_and_cards.away_fouls_against) / 2 )
fouls_and_cards = fouls_and_cards[['team', 'fouls_per_game', 'yellow_cards_per_game' , 'red_cards_per_game']].round(2)


# In[112]:


fouls_and_cards.head()


# In[113]:


# Most rude teams:
fouls_and_cards.sort_values('fouls_per_game',ascending = False).head().set_index('team')


# In[114]:


ax = sns.scatterplot(data = fouls_and_cards, x = 'fouls_per_game', y = 'yellow_cards_per_game', hue='team', 
size="fouls_per_game",sizes=(50, 500)).set_title("Fouls and cards\n")
plt.legend(bbox_to_anchor=(1.14, 0.5), loc='right', borderaxespad=0)

sns.despine()


# In[115]:


# assume that fouls_and_cards is a pandas DataFrame containing the data for the chart
fig, ax = plt.subplots()
sns.scatterplot(data=fouls_and_cards, x='fouls_per_game', y='yellow_cards_per_game', hue='team', size="fouls_per_game", sizes=(50, 500), ax=ax)
ax.set_title("Fouls and cards")
ax.legend(bbox_to_anchor=(1.14, 0.5), loc='right', borderaxespad=0)

sns.despine()

# display the plot in the Streamlit app
st.pyplot(fig)


# In[116]:


fouls_and_cards[['team', 'fouls_per_game', 'yellow_cards_per_game']] \
    .set_index('team') \
    .plot(kind='bar', title = 'Fouls and Yellow cards per game\n')
plt.show()


# In[117]:


# assume that fouls_and_cards is a pandas DataFrame containing the data for the chart
fig, ax = plt.subplots()
fouls_and_cards[['team', 'fouls_per_game', 'yellow_cards_per_game']].set_index('team').plot(kind='bar', ax=ax)
ax.set_title('Fouls and Yellow cards per game')

# display the plot in the Streamlit app
st.pyplot(fig)


# *Saudi Arabia and Serbia are leaders in a number of yellow cards per game.*
# 
# *Equador players foul a lot, but the number of yellow cards lower than mean value*
# 
# *Players from England, Germany and Spain are most friendly to their opponents*

# In[118]:


pressure = df_all[['team1', 'team2', 'forced_turnovers_team1', 'forced_turnovers_team2',
       'defensive_pressures_applied_team1',
       'defensive_pressures_applied_team2']]
pressure.head()


# In[119]:


home_pressure = pressure \
    .groupby('team1',as_index=False) \
    .agg({"forced_turnovers_team1":"mean",
          "defensive_pressures_applied_team1":"mean"}) \
    .rename(columns={"team1":"team",
                     "forced_turnovers_team1":"home_forced_turnovers",
                    "defensive_pressures_applied_team1":"home_defensive_pressures_applied"})
away_pressure = pressure \
    .groupby('team2',as_index=False) \
    .agg({"forced_turnovers_team2":"mean",
          "defensive_pressures_applied_team2":"mean"}) \
    .rename(columns={"team2":"team",
                     "forced_turnovers_team2":"away_forced_turnovers",
                    "defensive_pressures_applied_team2":"away_defensive_pressures_applied"})


# In[120]:


total_pressure = home_pressure.merge(away_pressure, on = 'team').copy().round(2)
total_pressure.head()


# In[121]:


total_pressure = total_pressure \
    .assign(forced_turnovers_per_game = \
            (total_pressure.home_forced_turnovers + total_pressure.away_forced_turnovers) / 2).round(2)
total_pressure = total_pressure \
    .assign(defensive_pressures_applied_per_game = \
            (total_pressure.home_defensive_pressures_applied + total_pressure.away_defensive_pressures_applied) / 2).round(2)
total_pressure = total_pressure.set_index('team')
total_pressure.head()


# In[122]:


total_pressure = total_pressure[['forced_turnovers_per_game', 'defensive_pressures_applied_per_game']] \
    .copy() \
    
total_pressure.sort_values('defensive_pressures_applied_per_game') \
    .plot(kind='bar', title = 'Defensive pressure and turnovers\n')
plt.show()


# In[123]:


total_pressure = total_pressure[['forced_turnovers_per_game', 'defensive_pressures_applied_per_game']].copy()
total_pressure.sort_values('defensive_pressures_applied_per_game').plot(kind='bar', title='Defensive pressure and turnovers')

# display the plot in the Streamlit app
st.pyplot(plt.gcf())


# In[124]:


sns.scatterplot(data = total_pressure.sort_values('defensive_pressures_applied_per_game', ascending=False), \
                x ='defensive_pressures_applied_per_game',
                y = 'forced_turnovers_per_game',hue = 'team',
                size="defensive_pressures_applied_per_game",
                sizes=(50, 500)) \
    .set_title("Defensive pressure and turnovers\n")

plt.legend(bbox_to_anchor=(1.3, 0.5), loc='right')

sns.despine()


# In[ ]:





# *We see that Japan, Costa Rika and Morocco feels the strongest pressure on themselves.*
# 
# *This is pretty logical, becouse this teams have lowest indexes of ball possession and mostly plays in defence position*

# In[125]:


final = df_all[['date', 'team1', 'team2', 'number_of_goals_team1',
               'number_of_goals_team2', 'total_attempts_team1', 'total_attempts_team2',
               'passes_team1', 'passes_team2']]
final.head()                            


# In[126]:


final_way_home = final.groupby(['team1','date'],as_index=False) \
    .agg({"number_of_goals_team1":"sum",
          "total_attempts_team1":"sum",
          "passes_team1":"sum"}) \
    .rename(columns={"team1":"team", 
                    "number_of_goals_team1":"number_of_goals",
                    "total_attempts_team1":"total_attempts",
                    "passes_team1":"passes"})
final_way_home = final_way_home.query("team =='ARGENTINA' | team == 'FRANCE'")
final_way_home


# In[127]:


final_way_away = final.groupby(['team2','date'],as_index=False) \
    .agg({"number_of_goals_team2":"sum",
          "total_attempts_team2":"sum",
          "passes_team2":"sum"}) \
    .rename(columns={"team2":"team", 
                    "number_of_goals_team2":"number_of_goals",
                    "total_attempts_team2":"total_attempts",
                    "passes_team2":"passes"})
final_way_away = final_way_away.query("team =='ARGENTINA' | team == 'FRANCE'")
final_way_away


# In[128]:


final_way = pd.concat([final_way_home, final_way_away]).sort_values('date').set_index('team')
final_way


# In[129]:


sns.scatterplot(data = final_way, x = 'date', y = 'passes', hue='team',
                size="total_attempts",
                sizes=(50, 500)) \
    .set_title("Passes and attempts in a way to final\n")
sns.despine()


# In[ ]:





# *We can notice that Argentina have better index of count of passes and attempts per game then France after finishing group stage.*
# 
# *As well Argentina had one more day to recovery between semi-final and final. May be it helped to accumulate last strength before the main game of 2022 World Cup*

# # STEP 4. Conclusion
# 
# 1. France and Argentina was the best in total goals and attempts realisation
# 2. Spain is absolut leader in passes, ball possession and accuracy, but if you want to win you also should score a lot.
# 3. Saudi Arabia and Serbia are leaders in a number of yellow cards per game. Spain and England were most friendly.
# 
# **The results of the national team of Argentina is explaind by**
# 
# * Good interactions between players
# * High index of attempts realisation 
# * Agressive game style with a lot of pressure on opponents defense
# * Low index of turnovers
# * Stable numbers during their way (except first game aggainst Saudi Arabia)
# 

# In[ ]:




