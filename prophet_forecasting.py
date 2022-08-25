#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# Import Libraries

# In[3]:
#def timeseries():

from array import array
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly
from fbprophet import Prophet


# Read DataSet

# In[48]:


df = pd.read_csv('C:/Users/shrey/FB-Prophet-Time-Series-Forecasting/MaunaLoaDailyTemps.csv')
df.dropna(inplace= True)
df.reset_index(drop=True, inplace=True)


# In[49]:


df.info()


# In[50]:


df.head()


# In[51]:


df=df[["DATE","AvgTemp"]]
df.head()


# # Change Column Names for FB Prophet

# In[52]:


df.columns = ['ds','y']


# In[53]:


df['ds'] = pd.to_datetime(df['ds'])
df.tail()


# # Plot Your Data

# In[10]:


df.plot(x='ds',y='y',figsize=(18,6))


# In[33]:


len(df)


# # Train, Test Split

# In[54]:


train = df.iloc[:len(df)-365]
test = df.iloc[len(df)-365:]


# # Start Making Predictions

# In[55]:


m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=365) #MS for monthly, H for hourly
forecast = m.predict(future)


# In[36]:


forecast.tail()


# In[56]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[38]:


test.tail()


# # USING BUILT-IN FB PROPHET VISUALIZATION
# #  

# In[19]:


#plot_plotly(m ,forecast)


# In[20]:


#plot_components_plotly(m, forecast)


# # Evaluate Your Model

# In[45]:


from statsmodels.tools.eval_measures import rmse


# In[43]:


predictions = forecast.iloc[-365:]['yhat']


# In[46]:


print("Root Mean Squared Error between actual and  predicted values: ",rmse(predictions,test['y']))
print("Mean Value of Test Dataset:", test['y'].mean())
print(predictions)
predictions.to_csv('hello.csv')


