#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet


# In[2]:


df = pd.read_csv('Ten-Year-Demand.csv')
m = Prophet(seasonality_mode='multiplicative').fit(df)
future = m.make_future_dataframe(periods=24, freq='M')
fcst = m.predict(future)
fig = m.plot(fcst)
plt.savefig('Forecast.png')


# In[3]:


m = Prophet(seasonality_mode='multiplicative', mcmc_samples=300).fit(df)
fcst = m.predict(future)
fig = m.plot_components(fcst)
plt.savefig('Forecast2.png')


# In[4]:


fcst.to_excel('Forecast.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:




