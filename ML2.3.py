#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


df=pd.read_csv('Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# # Grab X Values

#  'Percent of adults aged 18 years and older who have obesity'

# In[7]:


wrangled = df[df['StratificationID1'] == 'OVERALL'][['LocationDesc','Data_Value', 'Question', "YearStart" ]]
question = wrangled[wrangled['Question'] == 'Percent of adults who engage in no leisure-time physical activity'][['LocationDesc','Data_Value', 'Question', "YearStart" ]]
x_all = question[question['YearStart'] == 2014][['LocationDesc','Data_Value' ]]
x = question[question['YearStart'] == 2014][['Data_Value' ]].values
x


# # Grab Y Values

# Percent of adults who engage in no leisure-time physical activity'

# In[8]:


wrangled = df[df['StratificationID1'] == 'OVERALL'][['LocationDesc','Data_Value', 'Question', "YearStart" ]]
year = wrangled[wrangled['Question'] == 'Percent of adults aged 18 years and older who have obesity'][['LocationDesc','Data_Value', 'Question', "YearStart" ]]
y_all = year[year['YearStart'] == 2014][['LocationDesc','Data_Value' ]]
y = year[year['YearStart'] == 2014][['Data_Value' ]].values
y


# In[9]:


x = x.reshape(-1, 1)
x


# In[10]:


y = y.reshape(-1, 1)
y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, test_size=.2, random_state=100)


# In[14]:


print(f'X Train Data shape{x_train.shape}')
print(f'y Train Data shape{y_train.shape}')
print(f'X Test Data shape{x_test.shape}')
print(f'y Test Data shape{y_test.shape}')


# In[17]:


from matplotlib import pyplot as plt


# In[18]:


plt.scatter(x_train, y_train, color='orange')
plt.xlabel('% Adults with reporting no leisure Physical Activity')
plt.ylabel('% of Adults who have Obesity')
plt.title('Physical Data')
plt.show()


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


lm = LinearRegression()
lm.fit(x_train, y_train)
y_predict = lm.predict(x_test)


# In[22]:


print(f'Train Accuracy {round(lm.score(x_train, y_train)* 100,2)}%')
print(f'Test Accuracy {round(lm.score(x_test, y_test)* 100,2)}%')


# In[23]:


plt.scatter(x_train, y_train, color='orange')
plt.xlabel('% Adults with reporting no leisure Physical Activity')
plt.ylabel('% of Adults who have Obesity')
plt.title('Physical Data')
plt.show()


# In[ ]:




