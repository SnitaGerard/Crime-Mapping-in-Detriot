#!/usr/bin/env python
# coding: utf-8

# In[32]:


get_ipython().system('pip install opencage')
get_ipython().system('pip install folium')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install yellowbrick')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import folium
from folium.plugins import HeatMap
from opencage.geocoder import OpenCageGeocode
from wordcloud import WordCloud

#Preprocessing Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

# ML Libraries
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Evaluation Metrics
from yellowbrick.classifier import ClassBalance, ROCAUC, ClassificationReport, ClassPredictionError
from sklearn import metrics


# In[47]:


df = pd.read_csv(r'C:\Users\user\Downloads\RMS_Crime_Incidents.csv', index_col=None)


# In[48]:


df.head(5)


# In[21]:


df.describe().T


# In[22]:


df.dtypes


# In[23]:


df.shape


# In[49]:


df.drop(['crime_id','report_number','X','Y','oid','charge_description'], axis=1, inplace=True)


# In[50]:


df.head()


# In[51]:


df.columns = ['Address','Description','Offense','Offense_code','Arrest_Charge','Incident_timestamp','Time','Day','Hour','Year','Scout_car_area','Precinct','Block_id','Neighborhood','Council_dist','Zip','Long','Lat']
df.head()


# In[52]:


df['Incident_timestamp'] = pd.to_datetime(df['Incident_timestamp'])
df['Date'] = df['Incident_timestamp'].dt.date
df['Report_time'] = df['Incident_timestamp'].dt.time
df['Month'] = df['Incident_timestamp'].dt.month
df['Day_of_week'] = df['Incident_timestamp'].dt.day_name()
df['Day_number'] = df['Incident_timestamp'].dt.dayofweek


# In[38]:


df.head()


# In[53]:


df.drop(['Incident_timestamp', 'Day'], axis=1, inplace=True)


# In[90]:


df.head()


# In[54]:


df = df[['Address','Description','Offense','Offense_code','Report_time','Time','Day_number','Day_of_week','Hour','Month','Year','Date','Scout_car_area','Precinct','Block_id','Neighborhood','Council_dist','Zip','Long','Lat']]


# In[55]:


df.head()


# In[33]:


df.info()


# In[471]:


import datetime as dt


# In[36]:


# Create data for plotting
df['Day_of_year'] = df.Incident_timestamp.dt.dayofyear
data_holidays = df[df.Year == 2019].groupby(['Day_of_year']).size().reset_index(name='counts')

# Dates of major U.S. holidays in 2019
holidays = pd.Series(['2019-01-01', # New Years Day
                     '2019-01-21', # MLK Day
                     '2019-02-18', #presidents day
                     '2019-05-27', # Memorial Day
                     '2019-07-04', # Independence Day
                     '2019-09-02', # Labor Day
                     '2019-10-14', # Columbus day
                     '2019-11-11', # Veterans Day
                     '2019-11-28', # Thanksgiving
                     '2019-12-25']) # Christmas
holidays = pd.to_datetime(holidays).dt.dayofyear
holidays_names = ['NY',
                 'MLK',
                 'Presidents',
                 'Mem',
                 'July 4',
                 'Labor',
                 'C Day',
                 'Vet',
                 'Thx',
                 'Xmas']

import datetime as dt
# Plot crimes and holidays
fig, ax = plt.subplots(figsize=(15,10))
sns.lineplot(x='Day_of_year',
            y='counts',
            ax=ax,
            color='orange', 
            data=data_holidays)
plt.xlabel('Day of the year-2019', size=14)
plt.vlines(holidays, 0, 400, alpha=0.5, color ='purple', linewidth=2.5, linestyle=("solid"))
for i in range(len(holidays)):
    plt.text(x=holidays[i], y=250, s=holidays_names[i], size = 13)


# In[37]:


# Create data for plotting
df['Day_of_year'] = df.Incident_timestamp.dt.dayofyear
data_holidays = df[df.Year == 2020].groupby(['Day_of_year']).size().reset_index(name='counts')

# Dates of major U.S. holidays in 2017
holidays = pd.Series(['2020-01-01', # New Years Day
                     '2020-01-20', # MLK Day
                     '2020-02-17', #presidents day
                     '2020-05-25', # Memorial Day
                     '2020-07-04', # Independence Day
                     '2020-09-07', # Labor Day
                     '2020-10-12', # Columbus day
                     '2020-11-11', # Veterans Day
                     '2020-11-27', # Thanksgiving
                     '2020-12-25']) # Christmas
holidays = pd.to_datetime(holidays).dt.dayofyear
holidays_names = ['NY',
                 'MLK',
                 'Pres',
                 'Mem',
                 'July 4',
                 'Labor',
                 'C Day',
                 'Vet',
                 'Thx',
                 'Xmas']

import datetime as dt
# Plot crimes and holidays
fig, ax = plt.subplots(figsize=(15,10))
sns.lineplot(x='Day_of_year',
            y='counts',
            ax=ax,
            color='black', 
            data=data_holidays)
plt.xlabel('Day of the year-2020', size=14)
plt.vlines(holidays, 0, 400, alpha=0.5, color ='orange', linewidth=2.5, linestyle=("solid"))
for i in range(len(holidays)):
    plt.text(x=holidays[i], y=300, s=holidays_names[i], size = 13)


# In[53]:


df['Incident'] = 1


# In[54]:


df.head()


# In[16]:


df['Neighborhood'].value_counts()


# In[17]:


df['Council_dist'].value_counts()


# In[34]:


df['Offense'].value_counts()


# In[56]:


df['Offense'] = df['Offense'].str.replace('MISCELLANEOUS', 'OTHER')
df['Offense'] = df['Offense'].str.replace('LIQUOR', 'OUIL')
df['Offense'] = df['Offense'].str.replace('AGGRAVATED ASSAULT', 'ASSAULT')


# In[57]:


df['Offense'].value_counts()


# In[17]:


df.isnull().sum()


# In[58]:


df.shape


# In[21]:


df_cat = pd.pivot_table(df,
                       values=['Year'],
                       index=['Neighborhood'],
                       columns=['Offense'],
                       aggfunc=len,
                       fill_value=0,
                       margins=True)
df_cat


# In[22]:


df_cat.reset_index(inplace = True)
df_cat.columns = df_cat.columns.map(''.join)
df_cat.rename(columns={'YearAll':'Total'}, inplace=True)

df_cat.head()


# In[23]:


df_cat.sort_values(['Total'], ascending = False, axis = 0, inplace = True)
df_cat_top5 = df_cat.iloc[1:6]
df_cat_top5


# In[119]:


df_cat.describe().T


# In[124]:


df_dist = pd.pivot_table(df,
                       values=['Year'],
                       index=['Council_dist'],
                       columns=['Offense'],
                       aggfunc=len,
                       fill_value=0,
                       margins=True)
df_dist


# In[125]:


df_dist.reset_index(inplace = False)
df_dist.columns = df_dist.columns.map(''.join)
df_dist.rename(columns={'YearAll':'Total'}, inplace=True)

df_dist.head()


# In[128]:


df_dist.sort_values(['Total'], ascending = False, axis = 0, inplace = True)
df_dist_top3 = df_dist.iloc[1:4]
df_dist_top3


# In[129]:


df_dist.describe().T


# In[25]:


each_neighborhood = df_cat_top5[['Neighborhood', 'Total']]

each_neighborhood.set_index('Neighborhood', inplace = True)
ax = each_neighborhood.plot(kind='bar', color='Orange', figsize=(15, 10), rot=0)
ax.set_ylabel('Number of Crimes', fontsize = 14)
ax.set_xlabel('Neighbourhood', fontsize = 14)
ax.set_title('Neighbourhoods in Detroit with the Highest crimes')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14,
               )

plt.show()


# In[26]:


df_cat.sort_values(['Total'], ascending = True, axis = 0, inplace = True)
df_cat_low5 = df_cat.iloc[1:6]
df_cat_low5


# In[27]:


each_neighborhood = df_cat_low5[['Neighborhood', 'Total']]

each_neighborhood.set_index('Neighborhood', inplace = True)
ax = each_neighborhood.plot(kind='bar', color='Purple', figsize=(15, 10), rot=0)
ax.set_ylabel('Number of Crimes')
ax.set_xlabel('Neighbourhood')
ax.set_title('Neighbourhoods in Detroit with the Lowest crimes')

for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14,
               )

plt.show()


# In[206]:


plt.rcParams['figure.figsize'] = (20,9)
plt.style.use('bmh')
sns.countplot(df['Offense'], palette = 'gnuplot')


plt.xlabel('offense', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Major Crimes in Detroit', fontweight = 30, fontsize = 20)
plt.xticks(size=14, rotation = 90)
plt.yticks(size=14)

plt.show()


# In[428]:


plt.rcParams["figure.figsize"] = 15,10
order = df['Offense'].value_counts().head(5).index
sns.countplot(data = df, x = 'Offense', hue = 'Council_dist', order = order, palette="Set1")

plt.xlabel('offense', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Top 5 crimes - District Wise', fontweight = 30, fontsize = 20)
plt.xticks(size=14, rotation = 0)
plt.yticks(size=14)
plt.show()


# In[31]:


#mask = ((df['Year'] == 2017) | (df['Year'] == 2018) | (df['Year'] == 2019) | (df['Year'] == 2020))
grouped = df.groupby(['Month','Council_dist']).count()
sns.lineplot(data = grouped.reset_index(), x='Month', y='Offense',hue='Council_dist', palette="brg")
plt.xlabel('Month', fontsize=20)
plt.ylabel('Offense', fontsize=20)
plt.title('Crimes from 2013 - 2020', fontweight = 30, fontsize = 20)
plt.xticks(size=14, rotation = 0)
plt.yticks(size=14)
plt.show()


# In[32]:


mask = ((df['Year'] == 2017) | (df['Year'] == 2018) | (df['Year'] == 2019) | (df['Year'] == 2020))
grouped = df[mask].groupby(['Month','Council_dist']).count()
sns.lineplot(data = grouped.reset_index(), x='Month', y='Offense',hue='Council_dist', palette="brg")
plt.xlabel('Month', fontsize=20)
plt.ylabel('Offense', fontsize=20)
plt.title('Crimes from 2017 - 2020', fontweight = 30, fontsize = 20)
plt.xticks(size=14, rotation = 0)
plt.yticks(size=14)
plt.show()


# In[33]:


grouped = df.groupby(['Day_of_week','Council_dist']).count()
sns.FacetGrid(data = grouped.reset_index(), 
             palette = 'hsv',
             hue = "Day_of_week",
             height = 5).map(sns.kdeplot, "Offense", shade = True).add_legend();


# In[25]:


grouped = df.groupby(['Month','Council_dist']).count()
sns.FacetGrid(data = grouped.reset_index(), 
             palette = 'cool',
             hue = "Month",
             height = 5).map(sns.kdeplot, "Offense", shade = True).add_legend();


# In[34]:


graph = sns.countplot(data=df, x='Year', palette="Blues")

for p in graph.patches:
    graph.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14,
               )


plt.xlabel('Year', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Crimes from 2013-2020', fontweight = 30, fontsize = 20)
plt.xticks(size=14, rotation = 0)
plt.yticks(size=14)
plt.show()


# In[412]:


# Create a pivot table with month and category. 
dfPivYear = df.pivot_table(values='Incident', index='Offense', columns='Year', aggfunc=len)

fig, ax = plt.subplots(1, 1, figsize = (12, 6), dpi=300)
plt.title('Type of Crime By Year', fontsize=10)
plt.tick_params(labelsize=8)

sns.heatmap(
    dfPivYear.round(), 
    linecolor='lightgrey',
    linewidths=0.1,
    cmap='Reds', 
    annot=True, 
    fmt=".0f"
);

# Remove labels
ax.set_ylabel('Crime Type')    
ax.set_xlabel('Year')

plt.show()


# In[418]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Crime count by Category per year
dfPivCrimeDate = df.pivot_table(values='Incident'
                                     ,aggfunc=np.size
                                     ,columns='Offense'
                                     ,index='Year'
                                     ,fill_value=0)
plo = dfPivCrimeDate.plot(figsize=(20, 20), subplots=True, layout=(-1, 3), sharex=False, sharey=False)


# In[417]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Crime count by Category per year
dfPivCrime_hour = df.pivot_table(values='Incident'
                                     ,aggfunc=np.size
                                     ,columns='Offense'
                                     ,index='Hour'
                                     ,fill_value=0)
plo = dfPivCrime_hour.plot(figsize=(20, 20), subplots=True, layout=(-1, 3), sharex=False, sharey=False)


# In[35]:


months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
graph = sns.catplot(x='Month',
           kind='count',
            height=8, 
            aspect=3,
            color='Orange',
           data=df)
plt.xticks(np.arange(12), months, size=20)
plt.yticks(size=20)
plt.xlabel('')
plt.ylabel('Count', fontsize=15)


# In[37]:


plt.rcParams['figure.figsize'] = (20, 9)
plt.style.use('seaborn')

color = plt.cm.winter(np.linspace(0, 1, 15))
df['Address'].value_counts().head(15).plot.bar(color = color, figsize = (10, 8))

plt.title('Dangerous Streets of Detroit',fontsize = 14)

plt.xticks(rotation = 90)
plt.show()


# In[38]:


color = plt.cm.cool(np.linspace(0, 5, 100))
df['Time'].value_counts().head(20).plot.bar(color = color, figsize = (15, 9))

plt.title('Distribution of crime during the day', fontsize = 14)
plt.show()


# In[423]:


data = pd.crosstab(df['Offense'], df['Council_dist'])
color = plt.cm.gist_earth(np.linspace(0, 1, 10))

data.div(data.sum(1).astype(float), axis = 0).plot.bar(stacked = True, color = color, figsize = (12, 8))
plt.title('District vs Category of Crime', fontweight = 30, fontsize = 20)
plt.legend(loc='upper center', fontsize = 14, bbox_to_anchor=(1.10, 0.8), shadow=True, ncol=1)
plt.xlabel('offense', fontsize=20)
plt.ylabel('')
plt.xticks(size=14, rotation = 90)
plt.yticks(size=14)
plt.show()


# In[40]:


graph = sns.countplot(data=df, x='Council_dist', palette="Wistia")

for p in graph.patches:
    graph.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14,
               )


plt.xlabel('Council_dist', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Crimes-District wise', fontweight = 30, fontsize = 20)
plt.xticks(size=14, rotation = 0)
plt.yticks(size=14)
plt.show()


# In[43]:


df.Lat.replace(-1, None, inplace=True)
df.Long.replace(-1, None, inplace=True)

plt.rcParams["figure.figsize"] = 21,11

plt.subplots(figsize=(11,6))
sns.scatterplot(x='Lat',
                y='Long',
                palette='Set2',
                hue='Council_dist',
                alpha=0.1,
                data=df)
plt.legend(loc=2)


# In[23]:


dist_1=df.loc[df.Council_dist==6][df.Offense_code==2201][['Lat','Long']]
dist_1.Lat.fillna(0, inplace = True)
dist_1.Long.fillna(0, inplace = True) 

map_1=folium.Map(location=[42.35339333,-83.12672441], 
                 tiles = "OpenStreetMap",
                zoom_start=11)

folium.CircleMarker([42.33498839, -83.15450925],
                        radius=100,
                        fill_color="#b22222",
                        popup='Crimes',
                        color='red',
                       ).add_to(map_1)

HeatMap(data=dist_1, radius=16).add_to(map_1)

map_1


# In[364]:


df.info()


# In[59]:


# Convert Categorical Attributes to Numerical
df['Scout_car_area'] = pd.factorize(df["Scout_car_area"])[0]
df['Precinct'] = pd.factorize(df["Precinct"])[0]
df['Description'] = pd.factorize(df["Description"])[0]
df['Neighborhood'] = pd.factorize(df["Neighborhood"])[0]
df['Address'] = pd.factorize(df["Address"])[0] 


# In[60]:


Target = 'Offense'
print('Target:', Target)


# In[61]:


Classes = df['Offense'].unique()
Classes


# In[62]:


df['Offense'] = pd.factorize(df["Offense"])[0] 
df['Offense'].unique()


# In[17]:


X_fs = df.drop(['Offense'], axis=1)
Y_fs = df['Offense']

#Using Pearson Correlation
plt.figure(figsize=(20,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.xticks(size=20, rotation=90)
plt.yticks(size=20)
plt.show()


# In[63]:


Features = ["Offense_code", "Day_number", "Hour", "Month","Neighborhood","Council_dist"]
print('Full Features: ', Features)


# In[64]:


#Spliting dataset into Train Set & Test Set
x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]    #Features to train
x2 = x[Target]      #Target Class to train
y1 = y[Features]    #Features to test
y2 = y[Target]      #Target Class to test

print('Feature Set Used    : ', Features)
print('Target Class        : ', Target)
print('Training Set Size   : ', x.shape)
print('Test Set Size       : ', y.shape)


# In[44]:


from sklearn.linear_model import LogisticRegression


# In[54]:


lr_model = LogisticRegression(random_state = 5, solver = 'liblinear', multi_class='auto', max_iter=100)

lr_model.fit(X=x1,
             y=x2)

result = lr_model.predict(y[Features])


# In[70]:


# Model Evaluation
ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("========== Logistic Regression Results ==========")
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)


# In[58]:


# Classification Report
# Instantiate the classification model and visualizer
target_names = Classes
visualizer = ClassificationReport(lr_model, size=(1000, 720), classes=target_names)
visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
visualizer.score(y1, y2)       # Evaluate the model on the test data

print('================= Classification Report =================')
print('')
print(classification_report(y2, result, target_names=target_names))

g = visualizer.poof()  


# In[101]:


# Random Forest
# Create Model with configuration
rf_model = RandomForestClassifier(n_estimators=100, # Number of trees
                                  min_samples_split = 30,
                                  bootstrap = True, 
                                  max_depth = 50, 
                                  min_samples_leaf = 25)

# Model Training
rf_model.fit(X=x1,
             y=x2)

# Prediction
result2 = rf_model.predict(y[Features])


# In[111]:


# Model Evaluation
ac_sc2 = accuracy_score(y2, result2)
rc_sc2 = recall_score(y2, result2, average="weighted")
pr_sc2 = precision_score(y2, result2, average="weighted")
f1_sc2 = f1_score(y2, result2, average='micro')
confusion_m2 = confusion_matrix(y2, result2)

print("========== Random Forest Results ==========")
print("Accuracy    : ", ac_sc2)
print("Recall      : ", rc_sc2)
print("Precision   : ", pr_sc2)
print("F1 Score    : ", f1_sc2)
print("Confusion Matrix: ")
print(confusion_m2)


# In[105]:


# Classification Report
# Instantiate the classification model and visualizer
target_names = Classes
visualizer = ClassificationReport(rf_model, size=(1000, 720), classes=target_names)
visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
visualizer.score(y1, y2)       # Evaluate the model on the test data

print('================= Classification Report =================')
print('')
print(classification_report(y2, result2, target_names=target_names))

g = visualizer.poof()  


# In[17]:


# NN 

nn_model = MLPClassifier(solver='adam', 
                         alpha=1e-5,
                         hidden_layer_sizes=(150), 
                         random_state=1,
                         max_iter=300                         
                        )

# Model Training
nn_model.fit(X=x1,
             y=x2)

# Prediction
result1 = nn_model.predict(y[Features]) 


# In[18]:


# Model Evaluation
ac_sc1 = accuracy_score(y2, result1)
rc_sc1 = recall_score(y2, result1, average="weighted")
pr_sc1 = precision_score(y2, result1, average="weighted")
f1_sc1 = f1_score(y2, result1, average='micro')
confusion_m1 = confusion_matrix(y2, result1)

print("========== Neural Network Results ==========")
print("Accuracy    : ", ac_sc1)
print("Recall      : ", rc_sc1)
print("Precision   : ", pr_sc1)
print("F1 Score    : ", f1_sc1)
print("Confusion Matrix: ")
print(confusion_m1)


# In[24]:


target_names = Classes
visualizer = ClassificationReport(nn_model, size=(1000,750), classes=target_names)
visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
visualizer.score(y1, y2)       # Evaluate the model on the test data

print('================= Classification Report =================')
print('')
print(classification_report(y2, result1, target_names=target_names))

g = visualizer.poof() 


# In[66]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[67]:


DT_gini_model = DecisionTreeClassifier(criterion = "gini", random_state = 5,
                               max_depth=5, min_samples_leaf=8)


DT_gini_model.fit(X=x1,
             y=x2)

result3 = DT_gini_model.predict(y[Features])


# In[51]:


target_names=Classes
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=300)
tree.plot_tree(DT_gini_model,
               feature_names = Features, 
               class_names=target_names,
               filled = True);


# In[68]:


# Model Evaluation
ac_sc3 = accuracy_score(y2, result3)
rc_sc3 = recall_score(y2, result3, average="weighted")
pr_sc3 = precision_score(y2, result3, average="weighted",zero_division='warn')
f1_sc3 = f1_score(y2, result3, average='micro')
confusion_m3 = confusion_matrix(y2, result3)

print("========== Decision Tree Results ==========")
print("Accuracy    : ", ac_sc3)
print("Recall      : ", rc_sc3)
print("Precision   : ", pr_sc3)
print("F1 Score    : ", f1_sc3)
print("Confusion Matrix: ")
print(confusion_m3)


# In[69]:


target_names = Classes
visualizer = ClassificationReport(DT_gini_model, size=(1000,720), classes=target_names)
visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
visualizer.score(y1, y2)       # Evaluate the model on the test data

print('================= Classification Report =================')
print('')
print(classification_report(y2, result3, target_names=target_names))

g = visualizer.poof()


# In[21]:


# K-Nearest Neighbors
# Create Model with configuration 
knn_model = KNeighborsClassifier(n_neighbors=3)

# Model Training
knn_model.fit(X=x1,
             y=x2)

# Prediction
result4 = knn_model.predict(y[Features]) 


# In[22]:


# Model Evaluation
ac_sc4 = accuracy_score(y2, result4)
rc_sc4 = recall_score(y2, result4, average="weighted")
pr_sc4 = precision_score(y2, result4, average="weighted")
f1_sc4 = f1_score(y2, result4, average='micro')
confusion_m4 = confusion_matrix(y2, result4)

print("========== K-Nearest Neighbors Results ==========")
print("Accuracy    : ", ac_sc4)
print("Recall      : ", rc_sc4)
print("Precision   : ", pr_sc4)
print("F1 Score    : ", f1_sc4)
print("Confusion Matrix: ")
print(confusion_m4)


# In[24]:


# Classification Report
# Instantiate the classification model and visualizer
target_names = Classes
visualizer = ClassificationReport(knn_model, size=(1000,750), classes=target_names)
visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
visualizer.score(y1, y2)       # Evaluate the model on the test data

print('================= Classification Report =================')
print('')
print(classification_report(y2, result4, target_names=target_names))

g = visualizer.poof()


# In[20]:


from sklearn.ensemble import GradientBoostingClassifier


# In[21]:


#Gradient Boosting
# Create Model with configuration 
gbc_model = GradientBoostingClassifier(learning_rate=0.1, n_estimators = 70, random_state = 42)

# Model Training
gbc_model.fit(X=x1,
             y=x2)

# Prediction
result5 = gbc_model.predict(y[Features]) 


# In[22]:


# Model Evaluation
ac_sc5 = accuracy_score(y2, result5)
rc_sc5 = recall_score(y2, result5, average="weighted")
pr_sc5 = precision_score(y2, result5, average="weighted")
f1_sc5 = f1_score(y2, result5, average='micro')
confusion_m5 = confusion_matrix(y2, result5)

print("============= Gradient Boosting Results =============")
print("Accuracy    : ", ac_sc5)
print("Recall      : ", rc_sc5)
print("Precision   : ", pr_sc5)
print("F1 Score    : ", f1_sc5)
print("Confusion Matrix: ")
print(confusion_m5)


# In[23]:


target_names = Classes
visualizer = ClassificationReport(gbc_model,size=(1080, 720), classes=target_names)
visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
visualizer.score(y1, y2)       # Evaluate the model on the test data

print('================= Classification Report =================')
print('')
print(classification_report(y2, result5, target_names=target_names))

g = visualizer.poof()


# In[ ]:




