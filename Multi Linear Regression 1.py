#!/usr/bin/env python
# coding: utf-8

# # Multi Linear Regression Model-1

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv("ToyotaCorolla.csv",encoding ="latin1")
data


# ## EDA

# In[3]:


data.info()


# In[4]:


data.isna().sum()


# In[5]:


data=pd.concat([data.iloc[:,2:4],data.iloc[:,6:7],data.iloc[:,8:9],data.iloc[:,12:14],data.iloc[:,15:18]],axis=1)
data


# In[6]:


data=data.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
data


# In[7]:


data[data.duplicated()]


# In[8]:


data=data.drop_duplicates().reset_index(drop=True)
data


# In[9]:


data.describe()


# ## Correlation 

# In[10]:


data.corr()


# ## Scatterplot between variables along with histograms

# In[11]:


#Format the plot background and scatter plots for all the variables
sns.set_style(style='darkgrid')
sns.pairplot(data)


# ## Model Building

# In[12]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=data).fit()


# ## Model Testing

# In[13]:


model.params


# In[14]:


# Finding tvalues and pvalues
print(model.tvalues, '\n', model.pvalues)


# In[14]:


#R squared values
(model.rsquared,model.rsquared_adj)


# ## Simple Linear Regression Model

# In[15]:


slr_c =smf.ols('Price~CC',data = data).fit()  
print(slr_c.tvalues, '\n', slr_c.pvalues) 
slr_c.summary()


# In[16]:


slr_d =smf.ols('Price~Doors',data = data).fit()  
print(slr_d.tvalues, '\n', slr_d.pvalues)
slr_d.summary()


# In[17]:


mlr_cd=smf.ols('Price~CC+Doors',data=data).fit()
mlr_cd.tvalues , mlr_cd.pvalues # CC & Doors have significant pvalue
mlr_cd.summary()


# ## Model Validation Techniques
# ### Two Techniques: 1. Collinearity Check 

# In[18]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_age=smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=data).fit().rsquared
vif_age=1/(1-rsq_age)

rsq_KM=smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=data).fit().rsquared
vif_KM=1/(1-rsq_KM)

rsq_HP=smf.ols('HP~Age+KM+CC+Doors+Gears+QT+Weight',data=data).fit().rsquared
vif_HP=1/(1-rsq_HP)

rsq_CC=smf.ols('CC~Age+KM+HP+Doors+Gears+QT+Weight',data=data).fit().rsquared
vif_CC=1/(1-rsq_CC)

rsq_DR=smf.ols('Doors~Age+KM+HP+CC+Gears+QT+Weight',data=data).fit().rsquared
vif_DR=1/(1-rsq_DR)

rsq_GR=smf.ols('Gears~Age+KM+HP+CC+Doors+QT+Weight',data=data).fit().rsquared
vif_GR=1/(1-rsq_GR)

rsq_QT=smf.ols('QT~Age+KM+HP+CC+Doors+Gears+Weight',data=data).fit().rsquared
vif_QT=1/(1-rsq_QT)

rsq_WT=smf.ols('Weight~Age+KM+HP+CC+Doors+Gears+QT',data=data).fit().rsquared
vif_WT=1/(1-rsq_WT)

# Putting the values in Dataframe format
d1={'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],
    'Vif':[vif_age,vif_KM,vif_HP,vif_CC,vif_DR,vif_GR,vif_QT,vif_WT]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[19]:


# None variable has VIF>20, No Collinearity, so consider all varaibles in Regression


# ## 2. Residual Analysis¶
# ### Test for Normality of Residuals (Q-Q Plot)

# In[20]:


import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[21]:


list(np.where(model.resid>6000))  # outliar detection from above QQ plot of residuals


# In[22]:


list(np.where(model.resid<-6000))


# ##  Test for Homoscedasticity or Heteroscedasticity

# In[23]:


def standard_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[24]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 


# ## Residual Vs Regressors

# In[25]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)
plt.show()


# In[26]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "KM", fig=fig)
plt.show()


# In[27]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "HP", fig=fig)
plt.show()


# In[28]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "CC", fig=fig)
plt.show()


# In[29]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Doors", fig=fig)
plt.show()


# In[30]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Gears", fig=fig)
plt.show()


# In[31]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "QT", fig=fig)
plt.show()


# In[32]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Weight", fig=fig)
plt.show()


# ## Model Deletion Diagnostics (checking Outliers or Influencers)
# ### Two Techniques : 1. Cook's Distance

# In[33]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance


# In[34]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[35]:


#index and value of influencer where c is more than .5
(np.argmax(c),np.max(c))


# ### 2. Leverage value

# In[36]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[37]:


data.shape


# In[38]:


k = data.shape[1]
n = data.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[39]:


data[data.index.isin([80])]


# ## Improving the Model

# In[40]:


# Creating a copy of data so that original dataset is not affected
data1=data.copy()
data1


# In[41]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
data2=data1.drop(data1.index[[80]],axis=0).reset_index(drop=True)
data2


# In[42]:


final_data =smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=data2).fit() 
final_data.summary()


# ## Predicting for new Data

# In[43]:


# say New data for prediction is
new_data=pd.DataFrame({'Age':12,"KM":40000,"HP":80,"CC":1300,"Doors":4,"Gears":5,"QT":69,"Weight":1012},index=[0])
new_data


# In[44]:


# Manual Prediction of Price
final_data.predict(new_data)


# In[45]:


pred_y=final_data.predict(data2)
pred_y


# ### Table containing R^2 value for each prepared model

# In[47]:


d2={'Prep_Models':['Model','Final_Model'],'Rsquared':[model.rsquared,final_data.rsquared]}
table=pd.DataFrame(d2)
table


# In[ ]:




